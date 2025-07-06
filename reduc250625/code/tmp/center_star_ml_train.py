import os
import pandas as pd
import numpy as np

def extract_features(pos_csvnm, header):
    df = pd.read_csv(pos_csvnm)
    naxis1, naxis2 = int(header['NAXIS1']), int(header['NAXIS2'])
    center = np.array([naxis1 / 2, naxis2 / 2])
    coords = df[['xcent', 'ycent']].values
    dists = np.linalg.norm(coords - center, axis=1)
    features = pd.DataFrame({
        'obj_id': df['obj_id'],
        'dist_center': dists,
        'flux': df['flux'] if 'flux' in df.columns else np.nan,
        'peak': df['peak'] if 'peak' in df.columns else np.nan,
        # 可加更多特征
    })
    return features

# 构建训练集
import sys
from astropy.io import fits as pyfits

fit_path = '/home/vxpp/program/data/L2/imacal_vt/BD+28D4211/2024-11-19'
choice = pd.read_csv('center_choice.csv')
X, y = [], []
for _, row in choice.iterrows():
    img = row['img']
    obj_id = row['obj_id']
    pos_csvnm = os.path.splitext(img)[0] + '_pos.csv'
    if not os.path.exists(pos_csvnm):
        continue
    header = pyfits.getheader(os.path.join(fit_path, img))
    feats = extract_features(pos_csvnm, header)
    feats['label'] = (feats['obj_id'] == obj_id).astype(int)
    X.append(feats)
X = pd.concat(X, ignore_index=True)
y = X['label']
X = X.drop(['obj_id', 'label'], axis=1)
X = X.fillna(X.mean())

# 训练模型
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

import joblib
joblib.dump(clf, 'center_star_rf.joblib')
print("模型已保存为 center_star_rf.joblib")
