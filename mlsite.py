import streamlit as st
import folium
import geopandas as gpd
from streamlit_folium import folium_static
import pandas as pd
from folium.features import GeoJson
from PIL import Image
import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# 모델 불러오기

# Streamlit 앱에서 MinMaxScaler 불러오기
scaler = joblib.load('minmax_scaler.pkl')
model = joblib.load('linear_regression_model.pkl')

# shp 파일 읽기
gdf = gpd.read_file("./shp/표준단위구역.shp")
eda_data = pd.read_csv('visual_data.csv')
train_dataset = pd.read_csv('train_dataset.csv')
data = eda_data.copy()
# eda_data의 'SUD_CD' 열을 문자열 형식으로 변환
eda_data['SUD_CD'] = eda_data['SUD_CD'].astype(str)
new_data = pd.read_csv('mlsite_data.csv')
new_data_describe = new_data.describe()
# 변환된 eda_data의 'SUD_CD' 열에 있는 코드들만 다시 필터링
gdf_filtered = gdf[gdf['SUD_CD'].isin(eda_data['SUD_CD'])]
gdf = gdf_filtered.to_crs(epsg=4326)

index_code = '''Area = index[['SUD_CD','RELM_AR']]
Step_Zero = SURV_RT_QU[['STDR_YYQU_CD','SUD_CD','SVC_INDUTY_CD','YEAR_3_BEING_CO','YEAR_3_OPBIZ_STOR_CO']]
Step_Zero = Step_Zero.merge(Area, on ='SUD_CD')
SELNG_QU_tempt = SELNG_QU[['STDR_YYQU_CD','SUD_CD','SVC_INDUTY_CD','THSMON_SELNG_AMT']]
Step_One = Step_Zero.merge(SELNG_QU_tempt, on = ['STDR_YYQU_CD','SUD_CD','SVC_INDUTY_CD'])'''

boxenplot_code = '''
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# Standardize the data (excluding 'SUD_CD' and 'THSMON_SELNG_AMT')
tempt = train_dataset.drop(['STDR_YYQU_CD','SVC_INDUTY_CD','SUD_CD'],axis=1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(tempt)
scaled_df = pd.DataFrame(scaled_data, columns=tempt.columns)

# Create the boxplot
plt.figure(figsize=(15, 10))
sns.boxenplot(data=scaled_df)

# Set the title and labels in English
plt.title('Box Plot of APT')
plt.xticks(rotation=45)
plt.ylabel('Scaled Values')

# Show the plot
plt.show()'''

heatmap_code = '''tempt = train_dataset.drop('SVC_INDUTY_CD', axis=1).fillna(0)
correlation_matrix = tempt.corr()
plt.figure(figsize=(50, 20))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('corr Heatmap')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()'''

make_traindata_code = """eda_data = train_dataset[train_dataset['SVC_INDUTY_CD']=='CS100010']
eda_data = eda_data[eda_data['YEAR_3_OPBIZ_STOR_CO'] != 0]
eda_data = eda_data.drop(['STDR_YYQU_CD','SUD_CD','SVC_INDUTY_CD'],axis=1).fillna(0)"""

xgb_result = """| Error Type              | Value                 |
|-------------------------|-----------------------|
| Mean Squared Error      | 0.15777604395829098  |
| Mean Absolute Error     | 0.305254979683311   |
| Root Mean Squared Error | 0.39721032710428233    |
| R^2 Score               | 0.5429684942770547    |
"""

gbr_result = """| Error Type              | Value                 |
|-------------------------|-----------------------|
| Mean Squared Error      | 0.003519115422518033  |
| Mean Absolute Error     | 0.2105229266257965   |
| Root Mean Squared Error | 0.05932213265315091    |
| R^2 Score               | 0.9902599862811429    |
"""
gru_training_loss = """
Epoch 1/400
\n\n22/22 [==============================] - 1s 13ms/step - loss: 0.7688 - val_loss: 0.5182
\n\nEpoch 2/400
\n\n22/22 [==============================] - 0s 5ms/step - loss: 0.4369 - val_loss: 0.3796
\n\nEpoch 3/400
\n\n22/22 [==============================] - 0s 6ms/step - loss: 0.3863 - val_loss: 0.3403
\n\nEpoch 4/400
\n\n22/22 [==============================] - 0s 6ms/step - loss: 0.3744 - val_loss: 0.3396
\n\nEpoch 5/400
\n\n22/22 [==============================] - 0s 5ms/step - loss: 0.3636 - val_loss: 0.3400
\n\nEpoch 6/400
\n\n22/22 [==============================] - 0s 5ms/step - loss: 0.3513 - val_loss: 0.3625
\n\n...
\n\nEpoch 399/400
\n\n22/22 [==============================] - 0s 5ms/step - loss: 0.0032 - val_loss: 0.0096
\n\nEpoch 400/400
\n\n22/22 [==============================] - 0s 5ms/step - loss: 0.0032 - val_loss: 0.0049
"""

gru_result = """| Error Type              | Value                 |
|-------------------------|-----------------------|
| Mean Squared Error      | 0.023680935904595305  |
| Mean Absolute Error     | 0.04118206033398255   |
| Root Mean Squared Error | 0.15388611342351624    |
| R^2 Score               | 0.934457210721126    |
"""

lr_result = """| Error Type              | Value                 |
|-------------------------|-----------------------|
| Mean Squared Error      | 0.023642028574025222  |
| Mean Absolute Error     | 0.10233977797061283   |
| Root Mean Squared Error | 0.1537596454666348    |
| R^2 Score               | 0.9223344400860514    |
"""
lr_train_code = """X = new_data.drop('YEAR_3_BEING_CO', axis=1).values
y = new_data['YEAR_3_BEING_CO'].values
X.shape, y.shape
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
lg = LinearRegression()
lg.fit(X_train, y_train)
lg.score(X_test, y_test)
lg_pred = lg.predict(X_test)
r2 = r2_score(y_test, lg_pred)
mse = mean_squared_error(y_test, lg_pred)
mae = mean_absolute_error(y_test, lg_pred)
rmse = np.sqrt(mse)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print("R^2 Score:", r2)
"""
lr_visual_code = """import numpy as np
import matplotlib.pyplot as plt

features_mean = new_data.drop('YEAR_3_BEING_CO', axis=1).sample(n=1).mean()
similr_values = np.arange(0, 20)
predicted_values = []

for value in similr_values:

    data_point = features_mean.copy()
    data_point['SIMILR_INDUTY_STOR_CO'] = value


    data_point_scaled = scaler.transform([data_point])


    predicted = lg.predict(data_point_scaled)
    predicted_values.append(predicted[0])


plt.figure(figsize=(10, 6))
plt.plot(similr_values, predicted_values, marker='o')
plt.xlabel('SIMILR_INDUTY_STOR_CO')
plt.ylabel('Predicted YEAR_3_BEING_CO')


plt.title('Predicted YEAR_3_BEING_CO by SIMILR_INDUTY_STOR_CO')
plt.grid(True)
plt.show()
"""

xgb_train_code = """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import numpy as np

# 이미 준비된 데이터 사용
X = new_data.drop('YEAR_3_BEING_CO', axis=1).values
y = new_data['YEAR_3_BEING_CO'].values

# 훈련 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 데이터 스케일링
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# XGBoost 모델 생성 및 훈련
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train_scaled, y_train)

# 테스트 데이터에 대한 예측 수행
xg_pred = xg_reg.predict(X_test_scaled)

# 성능 평가
xg_mse = mean_squared_error(y_test, xg_pred)
xg_mae = mean_absolute_error(y_test, xg_pred)
xg_rmse = np.sqrt(xg_mse)

xg_r2 = r2_score(y_test, xg_pred)

print("Mean Squared Error:", xg_mse)
print("Mean Absolute Error:", xg_mae)
print("Root Mean Squared Error:", xg_rmse)
print("R2 score Error:", xg_r2)"""

gbr_train_code = """
X = new_data.drop(['YEAR_3_BEING_CO'], axis=1)
y = new_data['YEAR_3_BEING_CO']

# 훈련 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# 그라디언트 부스팅 모델 생성
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 훈련
gbr.fit(X_train, y_train)
# 테스트 데이터에 대한 예측 수행
y_pred = gbr.predict(X_test)

# 성능 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print("Mean Absolute Error:", xg_mae)
print("Root Mean Squared Error:", rmse)
print(f'R^2 Score: {r2}')"""

gru_train_code = """import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 데이터셋 준비 (당신의 코드에서 사용된 것과 같은 방식으로)
X = new_data.drop(['YEAR_3_BEING_CO'], axis=1)
y = new_data['YEAR_3_BEING_CO']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 데이터 정규화
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 모델 구축
model = Sequential()
model.add(GRU(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error')

# 모델 훈련
history = model.fit(X_train, y_train, epochs=400, batch_size=32, validation_split=0.2)
y_pred = model.predict(X_test)
# 훈련 손실과 검증 손실 추출
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 에포크 숫자 생성
epochs = range(1, len(train_loss) + 1)

# 성능 평가
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
print(f'R^2 Score: {r2}')
# 손실 그래프 그리기
plt.figure()
plt.plot(epochs, train_loss, 'b-', label='Training loss') 
plt.plot(epochs, val_loss, 'r-', label='Validation loss') 
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()"""

page1 = """# 프로젝트 중간 발표


## 소상공인을 위한 적정 카페 수 예측모델
### - 상관관계 분석과 머신러닝 모델을 중심으로

### 조원
- **강경헌: 수학과**
- **김도원: 전자전기컴퓨터공학부**
- **임지인: 공간정보공학과**
- **전백찬: 도시공학과**
- **표기환: 전자전기컴퓨터공학부**


## Research Topic

### Research 1: 카페의 적정 밀도.
- 최근 카페 수의 급증에 따라, 새로 개업하는 카페의 시장 경쟁력을 파악하기 위하여, 단위 구역 별 카페의 적정 밀도를 파악하는 모델을 생성하여 소상공인의 카페 개업 판단에 도움을 준다. 

### Research 2: 가까운 미래, 예상되는 카페의 숫자
- 유동인구, 상주인구의 변화를 시계열에 따라 분석하여, 가까운 미래 카페의 수를 분석한다. 이를 바탕으로 해당 단위구역의 카페의 증감을 파악하여 신규 개업, 업종 변경 등 소상공인에게 다양한 선택의 폭을 제공한다. 

### Research 3: 카페와 상생하는 업종
- 카페의 집적이 한계에 도달한 지역에 대하여, 카페와 시너지 효과를 내는 업종을 사용자에게 추천한다. 또는 주어진 단위구역의 상생, 악생업종을 분석하여 카페의 신규 개업 판단에 도움을 준다.




# 소개

## 프로젝트 목표

프로젝트의 주요 목표는 서울신용보증재단에서 제공한 데이터를 효과적으로 활용하여 카페 산업에 대한 다양하고 심층적인 연구를 수행하고, 이를 토대로 사용자 인터페이스를 제작하여 현장 적용 가능한 가치 있는 정보를 제공한다.

### 1. 데이터 수집 및 전처리
- 서울신용보증재단에서 제공한 다양한 데이터를 분석하고, 모델링에 사용할 핵심 데이터를 선별하는 과정을 진행한다.
- 상관관계 분석을 통해 Target Data와 다른 데이터 간의 상관관계를 체계적으로 파악하여 모델링에 활용할 적절한 데이터를 선정한다.

### 2. 모델 구축
- **연구 1번 (카페의 적정 밀도):** **회귀 분석**을 통해 지역별 적정한 카페의 수를 예측하고, 산출된 모델을 기반으로 적정 밀도를 제시한다.
- **연구 2번 (미래의 카페 수 예측):** **시계열 분석**을 활용하여 미래의 카페 수를 예측하고, 이를 통해 향후 시장 동향을 시각적으로 제공한다.
- **연구 3번 (카페와 상생하는 업종):** **군집 분석**을 통해 카페와 상생하는 업종을 도출하고, 이들 간의 유리한 조합과 협력 관계를 시각화하여 제시한다.

### 3. 시각화 및 사용자 인터페이스 제공
- 완성된 모델을 기반으로 직관적이고 효과적인 사용자 인터페이스를 구현한다.
- 지도 기능을 활용하여 사용자가 원하는 위치를 선택하면, 연구 과제 3가지 중 원하는 분석 내용을 선택할 수 있도록 한다.

> - <font color = 'green'> **카페의 적정 밀도 분석:**</font> 사용자가 선정한 구역의 카페 수를 조절하여 매출 및 생존율을 확인할 수 있도록 시뮬레이션을 제공한다.
> - <font color = 'green'> **장.단기 카페 수 예측:**</font> 시계열 분석 모델을 이용하여 선정 구역의 시간에 따른 카페 수의 변화 추이를 직관적으로 시각화합니다.
>- <font color = 'green'> **카페 상생업종 분석:**</font> 지역별 카페 포화 상태와 상생 업종의 종류, 이점을 지도와 차트를 활용하여 사용자에게 제시합니다.


"""

VF_score_code = '''from statsmodels.stats.outliers_influence import variance_inflation_factor

# VIF 계산을 위한 데이터 준비
# 'Unnamed: 0' 열도 제외 (보이는 것으로는 인덱스와 같은 역할을 할 것으로 보임)
vif_data = new_data.drop(['YEAR_3_BEING_CO'], axis=1)

# VIF 계산
vif = pd.DataFrame()
vif["Variable"] = vif_data.columns
vif["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]

vif.sort_values(by="VIF", ascending=False)  # VIF가 높은 순으로 정렬하여 출력

'''
VF_score_result = '''|    | Variable               | VIF         |
|----|------------------------|-------------|
| 19 | FRC_STOR_CO            | inf         |
| 16 | SIMILR_INDUTY_STOR_CO  | inf         |
| 15 | STOR_CO                | inf         |
| 14 | TOT_STOR_CO_90         | 214.333166  |
| 11 | TOT_STOR_CO            | 200.646316  |
| 13 | TOT_BSN_MT_90          | 181.556330  |
| 10 | TOT_BSN_MT             | 143.025113  |
| 6  | POPLTN_CO              | 77.387262   |
| 5  | HSHLD_INCOME_TOTAMT    | 75.236988   |
| 9  | SALE_MT_AVG            | 59.424699   |
| 12 | SALE_MT_AVG_90         | 55.250692   |
| 0  | YEAR_3_OPBIZ_STOR_CO   | 7.398432    |
| 4  | VIATR_FCLTY_CO         | 6.215402    |
| 8  | TOT_REPOP_CO           | 5.080620    |
| 1  | RELM_AR                | 3.488730    |
| 3  | YEAR_3_BEING_RT        | 3.335958    |
| 2  | THSMON_SELNG_AMT       | 2.562124    |
| 7  | RENT_CO                | 2.520136    |
| 17 | OPBIZ_STOR_CO          | 1.354334    |
| 18 | CLSBIZ_STOR_CO         | 1.276688    |
'''

# Streamlit 웹페이지 기본 설정
# 사이드바에 페이지 선택 옵션 추가
page = st.sidebar.selectbox(
    "페이지를 선택하세요",
    ["프로젝트 개요", "데이터 전처리", "모델 학습", "결과물", "향후 과제"]
)

# 페이지에 따른 내용 렌더링
if page == "프로젝트 개요":
    st.markdown(page1, unsafe_allow_html=True)
    # 로컬 이미지 파일 불러오기
    image1 = Image.open("./visual/222.JPG")
    # 이미지 표시
    st.image(image1, caption='이미지 설명')


elif page == "데이터 전처리":
    st.title("데이터 전처리")
    st.header("데이터 목록")
    markdown_table = """
    |      | 번호 | 테이블명(영문)          | 테이블명(한글)                   |
    |------|------|------------------------|---------------------------------|
    | 0    | 1    | 자치구영역             | 자치구영역                      |
    | 1    | 2    | 행정동영역             | 행정동영역                      |
    | 2    | 3    | 표준단위구역영역        | 표준단위구역영역                 |
    | 3    | 4    | 골목상권영역           | 골목상권영역                    |
    | 4    | 5    | 발달상권영역           | 발달상권영역                    |
    | 5    | 6    | 전통시장상권영역       | 전통시장상권영역                |
    | 6    | 6    | TBSM_SUD_APT_QU        | 아파트정보                      |
    | 7    | 7    | TBSM_SUD_BEING_RT_QU   | 연차별생존율_표준단위구역_분기   |
    | 8    | 8    | TBSM_SUD_FCLTY         | 표준단위구역집계_시설            |
    | 9    | 9    | TBSM_SUD_FLPOP_QU      | 표준단위구역집계_상존인구_분기   |
    | 10   | 10   | TBSM_SUD_HOUSINCOM_QU  | 표준단위구역집계_가구소득_분기   |
    | 11   | 11   | TBSM_SUD_RENT_CURPRC_QU| 표준단위구역집계_임대시세_분기   |
    | 12   | 12   | TBSM_SUD_REPOP_QU      | 표준단위구역집계_상주인구_분기   |
    | 13   | 13   | TBSM_SUD_SALE_MT_QU    | 표준단위구역집계 평균영업개월_분기|
    | 14   | 14   | TBSM_SUD_SELNG_QU      | 표준단위구역집계_매출_분기       |
    | 15   | 15   | TBSM_SUD_STOR_QU       | 표준단위구역_점포_개폐업_분기   |
    | 16   | 16   | TBSM_SUD_SURV_RT_QU    | 신생기업생존율_표준단위구역_분기 |
    | 17   | 17   | TBSM_SUD_WRC_POPLTN_QU | 표준단위구역집계_직장인구_분기   |
    """
    # Streamlit에서 마크다운 형식의 테이블 표시
    st.markdown(markdown_table)
    st.write("")
    st.write("""해당 데이터를 기반으로 RESEARCH 1. 카페의 적정 개수를 구하고, 새로운 카페를 창업했을 때 그 카페가 생존할 수 있는지 예측해보고자 하였다. 
             이를 위해서 TBSM_SUD_SURV_RT_QU의 ['YEAR_3_BEING_RT'] 변수를 TARGET으로 선정하였으며, 각 데이터의 FEATURE들과의 관계성을 확인해보고자 했다.""")
    st.markdown("---")
    st.header("INDEX 생성")
        # 로컬 이미지 파일 불러오기
    st.code(index_code,language='python')
    image2 = Image.open("./visual/2.PNG")
    # 이미지 표시
    st.image(image2, caption='표준단위구역코드를 기준으로 병합')
    st.markdown("##### 제공된 표준단위구역의 SHP파일에서 표준단위구역 코드와 면적을 기초 INDEX로 설정하였고, 3년차 생존율과 총매출을 시작으로 데이터를 병합하여 데이터의 특성을 확인하였다.")
    
    image3 = Image.open("./visual/면적,생존율,매출액.PNG")
    st.image(image3, caption='표준단위구역코드를 기준으로 병합')

    image4 = Image.open("./visual/3.PNG")
    st.image(image4, caption='기초 데이터셋 히트맵')

    image5 = Image.open("./visual/가구소득.png")
    st.image(image5, caption='가구소득과의 히트맵')

    image6 = Image.open("./visual/기반시설.PNG")
    st.image(image6, caption='기반시설과의 히트맵')

    image7 = Image.open("./visual/상존인구.PNG")
    st.image(image7, caption='상존인구와의 히트맵')

    image8 = Image.open("./visual/상주인구.PNG")
    st.image(image8, caption='상주인구와의 히트맵')

    image9 = Image.open("./visual/임대시세.PNG")
    st.image(image9, caption='임대시세와의 히트맵')

    image10 = Image.open("./visual/점포개폐업.PNG")
    st.image(image10, caption='점포개폐업와의 히트맵')

    image11 = Image.open("./visual/평균영업개월.PNG")
    st.image(image11, caption='평균영업개월과의 히트맵')
    st.markdown("---")
    st.markdown("##### 제공된 데이터셋과 표준단위구역의 신생기업생존율, 총 매출량 사이의 상관관계를 파악하여, 최종적인 데이터셋을 도출하였다.")

    markdown_table2 = """|    | Column                | Non-Null Count | KOREAN   |
|----|-----------------------|----------------|---------|
| 0  | STDR_YYQU_CD          | 103638 non-null| 기준_년분기_코드   |
| 1  | SUD_CD                | 103638 non-null| 표준단위구역_코드   |
| 2  | SVC_INDUTY_CD         | 103638 non-null| 서비스_업종_코드  |
| 3  | YEAR_3_BEING_CO       | 103638 non-null| 신생기업_3년_생존_수   |
| 4  | YEAR_3_OPBIZ_STOR_CO  | 103638 non-null| 신생기업_3년_개업_점포_수   |
| 5  | RELM_AR               | 103638 non-null| 영역_면적 |
| 6  | THSMON_SELNG_AMT      | 103638 non-null| 당월_매출_금액   |
| 7  | YEAR_3_BEING_RT       | 103638 non-null| 3년_생존_율 |
| 8  | VIATR_FCLTY_CO        | 103638 non-null| 집객시설_수   |
| 9  | HSHLD_INCOME_TOTAMT   | 103638 non-null| 가구소득_총금액   |
| 10 | POPLTN_CO             | 103638 non-null| 인구_수   |
| 11 | RENT_CO               | 103638 non-null| 임대건수   |
| 12 | TOT_REPOP_CO          | 103638 non-null| 총_상주인구_수   |
| 13 | SALE_MT_AVG           | 103638 non-null| 평균영업개월(10년전기준) |
| 14 | TOT_BSN_MT            | 103638 non-null| 총_영업개월(10년전기준)   |
| 15 | TOT_STOR_CO           | 103638 non-null| 총_점포_수(10년전기준)   |
| 16 | SALE_MT_AVG_90        | 103638 non-null| 평균영업개월(1990년기준) |
| 17 | TOT_BSN_MT_90         | 103638 non-null| 총_영업개월(1990년기준)   |
| 18 | TOT_STOR_CO_90        | 103638 non-null| 총_점포_수(1990년기준)   |
| 19 | STOR_CO               | 103638 non-null| 일반_점포_수   |
| 20 | SIMILR_INDUTY_STOR_CO | 103638 non-null| 총_점포_수   |
| 21 | OPBIZ_STOR_CO         | 103638 non-null| 개업_점포_수   |
| 22 | CLSBIZ_STOR_CO        | 103638 non-null| 폐업_점포_수   |
| 23 | FRC_STOR_CO           | 103638 non-null| 프랜차이즈_점포_수   |
"""
    st.code("train_dataset.info()", language='python')
    st.markdown(markdown_table2)
    st.write("인덱스 3개 feature 20개")
    st.code(boxenplot_code, language='python')
    image12 = Image.open("./visual/4.PNG")
    st.image(image12, caption='데이터셋의 Boxenplot')
    st.code(heatmap_code, language='python')
    image13 = Image.open("./visual/train_dataset_hitmap.PNG")
    st.image(image13, caption='데이터셋의 HeatMAP')

elif page == "모델 학습":
    st.title("모델 학습")
    st.subheader("전처리 데이터셋.iloc(:1000)")
    st.write(train_dataset)
    st.code(make_traindata_code, language='python')
    st.write("전처리 데이터셋에서 업종이 카페이고, 신생기업수가 != 0 인 행만 추출한뒤 인덱스값을 제거하여 학습을 위한 데이터셋을 제작")
    # Machine Learning 관련 내용 표시
    st.subheader("학습데이터셋 기초통계량")
    st.write(new_data_describe)
    image16 = Image.open("./visual/1111.png")
    st.image(image16, caption='타겟 데이터의 분포')
    st.code(VF_score_code, language='python')
    st.write(VF_score_result)
    st.write(''''FRC_STOR_CO' (프랜차이즈_점포_수), 'SIMILR_INDUTY_STOR_CO' (총_점포_수), 'STOR_CO' (일반_점포_수)의 VIF 값이 무한대로 나타남. 이는 이 변수들이 서로 완벽한 선형 관계에 있다는 것을 의미. 이 경우 한 두 개의 변수를 제거하는 것이 좋을 수 있습니다.
'TOT_STOR_CO_90' (총_점포_수(1990년기준)), 'TOT_STOR_CO' (총_점포_수(10년전기준)), 'TOT_BSN_MT_90' (총_영업개월(1990년기준)), 'TOT_BSN_MT' (총_영업개월(10년전기준)) 역시 높은 VIF 값을 보이고 있다. 따라서 다중공선성 문제가 있는 변수들을 제거하여 모델을 학습하였다. ''' 
)
    image19 = Image.open("./visual/traindata_heatmap.png")
    st.image(image19, caption='최종 학습데이터의 Heatmap')
    image20 = Image.open("./visual/traindata_boxenplot.png")
    st.image(image20, caption='최종 학습데이터의 boxenplot')

    st.subheader("선형회귀 머신러닝 모델 제작")
    st.code(lr_train_code, language='python')
    image14 = Image.open("./visual/선형회귀 학습결과.PNG")
    st.image(image14, caption='선형회귀모델 학습 및 결과')
    st.write(lr_result)
    st.write('R2 Score가 0.92로 설명력이 매우 높은 모델임을 확인할 수 있었다.')
    st.subheader("머신러닝 모델을 활용한 예측")
    st.code(lr_visual_code, language='python')
    st.subheader("특정 단위구역의 기존점포개수가 3년 생존 신생기업수에 미치는 영향")
    image15 = Image.open("./visual/선형회귀모델시각화.png")
    st.image(image15, caption='선형회귀모델을 이용한 무작위 표준단위구역에 대한 예측')
    st.write("기존 총점포수의 변수와 관계없이, 신생기업 입주시 3년 생존은 정해진 범위내에서 이루어지는 것을 확인 할 수 있음")
    st.subheader("특정 단위구역의 신생기업 수가 3년 생존 신생기업수에 미치는 영향")
    image19 = Image.open("./visual/선형회귀모델시각화2.png")
    st.image(image19, caption='선형회귀모델을 이용한 무작위 표준단위구역에 대한 예측')
    st.write("신생기업이 몇개일 때 3년 생존기업의 숫자를 예측할 수 있음")
    st.write("동일한 모델을 활용하여 면적, 밀도, 매출 등의 변화를 고려하여 적정 밀도를 구할 수 있을 것으로 예상")
    st.header("다양한 머신러닝 모델 학습")
    st.subheader("XGBoost 모델")
    st.code(xgb_train_code, language='python')
    st.write(xgb_result)
    st.subheader("GradientBoost 모델")
    st.code(gbr_train_code, language='python')
    st.write(gbr_result)
    st.header("딥러닝 모델 학습")
    st.subheader("GRU 모델")
    st.code(gru_train_code, language='python')

    image17 = Image.open("./visual/LOSS 그래프.png")
    st.image(image17, caption='EPOCH 1000')
    image18 = Image.open("./visual/GRURESULT.png")
    st.markdown('300~400 사이에서 과적합이 발생하는 것을 확인 가능')
    st.image(image18, caption='EPOCH 400')
    st.markdown('학습 과정')
    st.markdown(gru_training_loss)
    st.write(gru_result)
elif page == "결과물":
    st.title("표준단위구역 기반 카페 생존 예측")


    # 지도 초기화
    m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)

    # 표준단위구역 데이터 지도에 추가
    geojson = GeoJson(
        gdf.to_json(),
        name='표준단위구역',
        style_function=lambda x: {'fillColor': 'blue', 'color': 'black'},
        tooltip=folium.features.GeoJsonTooltip(fields=['SUD_CD'], aliases=['표준단위구역 코드:']),
    )
    m.add_child(geojson)

    # Streamlit에 지도 표시
    folium_static(m)

    # Streamlit 앱 타이틀
    st.title('창업을 희망하는 지역을 알려주세요')

    # 사용자 입력
    sud_cd = st.number_input('SUD_CD를 입력하세요', value=1129002290)

    # 데이터 필터링 및 표시
    filtered_data = data[data['SUD_CD'] == sud_cd]
    selected_columns = data.columns.drop(['YEAR_3_BEING_CO','SUD_CD'])
    if not filtered_data.empty:
        st.write('선택한 SUD_CD에 대한 데이터:', filtered_data)
        
        # 값 조정을 위한 인터페이스        # 범위 지정을 위한 슬라이더
        year_3_opbiz_stor_co_range = st.slider('신생기업 범위를 조정하세요', 0, 20, (0, 10))

        # 예측 버튼
        if st.button('예측하기'):
            # 필터링된 데이터에서 필요한 열 선택 및 값 조정
            #predict_data = filtered_data[selected_columns].copy()
            #predict_data['YEAR_3_OPBIZ_STOR_CO'] = year_3_opbiz_stor_co_range
            #predict_data['SIMILR_INDUTY_STOR_CO'] = similr_induty_stor_co_range

            # 데이터 스케일링 및 예측
            #predict_data_scaled = scaler.transform(predict_data)
            #prediction = model.predict(predict_data_scaled)
            
            # 예측 결과 표시
            image21 = Image.open("./visual/그래디언트부스트모델시각화2.png")
            st.image(image21, caption='신생기업 range[0-10]')
            st.write('선택한 지역의 카페 창업시 3년뒤 생존율은 다음과 같습니다.')
            st.write('선택한 지역의 적정한 카페의 수는 **최대 3개** 입니다. ')
            st.write('''| 신생기업수 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|-----------|---|---|---|---|---|---|---|---|---|----|
| 3년 뒤 생존수 | 1.0 | 2.0 | 2.5 | 3.0 | 3.0 | 3.5 | 3.5 | 3.5 | 3.5 | 3.5 |''')
    else:
        st.write('해당하는 SUD_CD 데이터가 없습니다.')


elif page == "향후 과제":
    st.title("향후 과제")
    
    # 사용자 입력
    sud_cd = st.number_input('SUD_CD를 입력하세요', value=1129002290)

    # 데이터 필터링 및 표시
    filtered_data = data[data['SUD_CD'] == sud_cd]
    selected_columns = data.columns.drop(['YEAR_3_BEING_CO', 'SUD_CD'])  # SUD_CD도 제외
    if not filtered_data.empty:
        st.write('선택한 SUD_CD에 대한 데이터:', filtered_data)
        
        # 값 조정을 위한 인터페이스
        similr_induty_stor_co = st.number_input('현재 총 점포수를 입력하세요', value=filtered_data.iloc[0]['SIMILR_INDUTY_STOR_CO'])

        # 범위 지정을 위한 슬라이더
        year_3_opbiz_stor_co_range = st.slider('신생기업 숫자 범위를 선택하세요', 0, 100, (10, 50))

        # 예측 및 그래프 버튼
        if st.button('예측 및 그래프 보기'):
            predicted_values = []
            range_values = np.arange(year_3_opbiz_stor_co_range[0], year_3_opbiz_stor_co_range[1] + 1)

            for value in range_values:
                predict_data = filtered_data[selected_columns].copy()
                predict_data['YEAR_3_OPBIZ_STOR_CO'] = value
                predict_data['SIMILR_INDUTY_STOR_CO'] = similr_induty_stor_co

                # 데이터 스케일링 및 예측
                predict_data_scaled = scaler.transform(predict_data)
                prediction = model.predict(predict_data_scaled)
                predicted_values.append(prediction[0])
            
            # 예측 결과 그래프 표시
            plt.figure(figsize=(10, 6))
            plt.plot(range_values, predicted_values, marker='o')
            plt.xlabel('YEAR_3_OPBIZ_STOR_CO')
            plt.ylabel('예측된 YEAR_3_BEING_CO')
            plt.title('예측된 YEAR_3_BEING_CO by YEAR_3_OPBIZ_STOR_CO')
            plt.grid(True)
            st.pyplot(plt)
