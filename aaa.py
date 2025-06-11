import streamlit as st
import sys
import pathlib
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from difflib import get_close_matches
from fastai.vision.all import load_learner, PILImage

# ------------------ Streamlit 页面配置 ------------------
st.set_page_config(page_title="综合景点推荐系统", layout="wide")

# ------------------ Python 版本检查 ------------------
if sys.version_info >= (3, 13):
    st.error("⚠️ 当前 Python 版本为 3.13+，可能与 fastai 不兼容。建议使用 Python 3.11。")
    st.stop()

# ------------------ Windows 路径兼容 ------------------
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

# ------------------ 模型与数据加载函数 ------------------
@st.cache_resource
def load_content_model():
    path = pathlib.Path(__file__).parent / "scenic_model.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_feature_data():
    excel = pd.ExcelFile('景点表.xlsx')
    df = excel.parse('Sheet1')
    return df

@st.cache_data
def load_scenic_info():
    df = pd.read_excel('景点表（新）.xlsx', dtype={'景点id': int})
    df = df.rename(columns={
        '景点id': 'scenic_id',
        '景点名称': 'scenic_name',
        '景点简介': 'scenic_introduction',
        '景点地址': 'scenic_address'
    })[['scenic_id', 'scenic_name', 'scenic_introduction', 'scenic_address']]

    def imgs(r):
        folder = os.path.join('图片', r.scenic_name)
        if os.path.exists(folder):
            return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png'))][:3]
        return []
    
    df['image_paths'] = df.apply(imgs, axis=1)
    return df

@st.cache_resource
def load_cls_model():
    temp = None
    if sys.platform == "win32":
        temp = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        model_path = pathlib.Path(__file__).parent / "风景.pkl"
        model = load_learner(model_path)
    except Exception as e:
        st.error(f"加载分类模型失败: {e}")
        st.stop()
    finally:
        if sys.platform == "win32" and temp is not None:
            pathlib.PosixPath = temp
    return model

# ------------------ 内容过滤数据准备 ------------------
model_data = load_content_model()
df_feat = load_feature_data()

if isinstance(model_data, dict):
    if '景点_id' in model_data and 'feature_vector' in model_data and 'feature_list' in model_data:
        scenic_ids = model_data['景点_id']
        feature_vectors = np.array(model_data['feature_vector'])
        feature_lists = model_data['feature_list']
    else:
        scenic_ids = model_data.get('scenic_ids', df_feat['景点id'].tolist())
        feature_vectors = np.array(
            model_data.get('feature_vectors', df_feat.drop(columns=['景点id', '景点名称', '简介', '地理位置']).to_numpy())
        )
        names = df_feat.drop(columns=['景点id', '景点名称', '简介', '地理位置']).columns.tolist()
        feature_lists = [[names[i] for i, val in enumerate(row) if val == 1] for row in feature_vectors]
else:
    scenic_ids = df_feat['景点id'].tolist()
    feature_vectors = df_feat.drop(columns=['景点id', '景点名称', '简介', '地理位置']).to_numpy()
    names = df_feat.drop(columns=['景点id', '景点名称', '简介', '地理位置']).columns.tolist()
    feature_lists = [[names[i] for i, val in enumerate(row) if val == 1] for row in feature_vectors]

all_features = sorted({f for feats in feature_lists for f in feats})

@st.cache_data
def calculate_similarity():
    if feature_vectors.size == 0 or feature_vectors.shape[1] == 0:
        return np.array([])
    norm = np.linalg.norm(feature_vectors, axis=1)
    if np.all(norm > 0):
        vecs = feature_vectors / norm[:, None]
        return cosine_similarity(vecs)
    return cosine_similarity(feature_vectors)

similarity_matrix = calculate_similarity()
scenic_id_to_idx = {sid: i for i, sid in enumerate(scenic_ids)}
scenic_name_to_id = {row['景点名称']: row['景点id'] for _, row in df_feat.iterrows()}

# ------------------ 工具函数 ------------------
def recognize_images(files, model):
    results = []
    for f in files:
        image = PILImage.create(f)
        pred, pred_idx, probs = model.predict(image)
        results.append((str(pred), probs[pred_idx].item()))
    return results

def fuzzy_match(name, names):
    m = get_close_matches(name, names, n=1, cutoff=0.6)
    return m[0] if m else name

# ------------------ 页面逻辑 ------------------
def page_select():
    st.header("📷 上传三张景点照片进行识别")
    files = st.file_uploader("上传三张图片", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

    if files:
        if len(files) < 3:
            st.error("⚠️ 请上传至少三张图片。")
            st.stop()

        model = load_cls_model()
        preds = recognize_images(files, model)
        df_info = load_scenic_info()

        st.subheader("🔍 识别结果与推荐景点：")
        cols = st.columns(3)
        for i, (pred_name, pred_score) in enumerate(preds):
            matched_name = fuzzy_match(pred_name, df_info['scenic_name'].tolist())
            row = df_info[df_info['scenic_name'] == matched_name]

            with cols[i]:
                st.image(files[i], caption=f"上传图 {i+1}", use_container_width=True)
                if not row.empty:
                    scenic = row.iloc[0]
                    st.markdown(f"**预测结果：{scenic['scenic_name']}**")
                    st.markdown(f"📈 准确率: `{pred_score:.2%}`")
                    st.markdown(f"🗺️ 地址: {scenic['scenic_address']}")
                    st.markdown(f"📖 简介: {scenic['scenic_introduction'][:100]}...")
                    for img in scenic['image_paths']:
                        st.image(img, width=120)
                else:
                    st.warning(f"⚠️ 未找到匹配的景点: `{pred_name}`")
    else:
        st.info("请上传三张图片。")

def page_content_single():
    st.header("🎯 内容过滤：基于单个景点推荐")
    selected = st.selectbox("选择一个景点：", df_feat['景点名称'].tolist())
    if selected and similarity_matrix.size:
        sid = scenic_name_to_id.get(selected)
        idx = scenic_id_to_idx.get(sid)
        if idx is not None:
            sims = similarity_matrix[idx].argsort()[::-1][1:11]
            for i, j in enumerate(sims):
                info = df_feat.iloc[j]
                score = similarity_matrix[idx][j]
                st.markdown(f"{i+1}. {info['景点名称']} (相似度: {score:.4f})")
                st.markdown(f"- 🎯 景点ID: {info['景点id']}")
                st.markdown(f"- 📌 地理位置: {info['地理位置']}")
                st.markdown(f"- 🌟 简介: {info['简介'][:100]}...\n---")
        else:
            st.warning("未能匹配到景点 ID。")
    else:
        st.info("请先选择一个景点。")

def page_content_attributes():
    st.header("🧩 内容过滤：基于属性组合推荐")
    selected = st.multiselect("选择一个或多个属性：", all_features)
    if selected:
        user_vec = np.zeros(len(all_features))
        for f in selected:
            if f in all_features:
                user_vec[all_features.index(f)] = 1
        sims = cosine_similarity([user_vec], feature_vectors)[0]
        idxs = sims.argsort()[::-1][:10]
        for i, j in enumerate(idxs):
            info = df_feat.iloc[j]
            score = sims[j]
            st.markdown(f"{i+1}. {info['景点名称']} (匹配度: {score:.4f})")
            st.markdown(f"- 🎯 景点ID: {info['景点id']}")
            st.markdown(f"- 📌 地理位置: {info['地理位置']}")
            st.markdown(f"- 🌟 简介: {info['简介'][:100]}...\n---")
    else:
        st.info("请至少选择一个属性。")

# ------------------ 主函数 ------------------
def main():
    st.title("🌄 综合景点推荐系统")
    tabs = st.tabs(["📷 图片识别推荐", "📍 单个景点推荐", "🧩 属性组合推荐"])
    with tabs[0]:
        page_select()
    with tabs[1]:
        page_content_single()
    with tabs[2]:
        page_content_attributes()

if __name__ == "__main__":
    main()
