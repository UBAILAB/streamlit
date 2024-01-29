import streamlit as st
import streamlit as st
import folium
import geopandas as gpd
from streamlit_folium import folium_static
import pandas as pd
from folium.features import GeoJson
import streamlit as st
from PIL import Image
import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler


menu = ["프로젝트 개요", "RESEARCH.1", "RESEARCH.2", "RESEARCH.3"]
choice = st.sidebar.selectbox("메뉴를 선택하세요", menu)

if choice == "프로젝트 개요":
    st.title("소상공인을 위한 인공지능 예측 연구 보고서")
    st.markdown('-----')
        
    st.header("프로젝트 개요")
    st.markdown("""


### 소상공인을 위한 적정 카페 수 예측모델
#### - 상관관계 분석과 머신러닝 모델을 중심으로

### 조원
- #### **강경헌: 수학과** | 딥러닝 모델 학습 및 알고리즘 설계
- #### **김도원: 전자전기컴퓨터공학부**  | 데이터 분석 및 엔지니어링
- #### **임지인: 공간정보공학과** | 공간 분석 및 데이터 시각화
- #### **전백찬: 도시공학과** | 머신러닝 모델학습 및 최적화
- #### **표기환: 전자전기컴퓨터공학부** | 군집분석 알고리즘 개발


## RESEARCH Topic

### RESEARCH 1: 카페의 적정 밀도.
- 최근 카페 수의 급증에 따라, 새로 개업하는 카페의 시장 경쟁력을 파악하기 위하여, 단위 구역 별 카페의 적정 밀도를 파악하는 모델을 생성하여 소상공인의 카페 개업 판단에 도움을 준다. 

### RESEARCH 2: 가까운 미래, 예상되는 카페의 숫자
- 유동인구, 상주인구의 변화를 시계열에 따라 분석하여, 가까운 미래 카페의 수를 분석한다. 이를 바탕으로 해당 단위구역의 카페의 증감을 파악하여 신규 개업, 업종 변경 등 소상공인에게 다양한 선택의 폭을 제공한다. 

### RESEARCH 3: 카페와 상생하는 업종
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
> - <font color = 'green'> **카페 상생업종 분석:**</font> 지역별 카페 포화 상태와 상생 업종의 종류, 이점을 지도와 차트를 활용하여 사용자에게 제시합니다.


""",unsafe_allow_html=True)


elif choice == "RESEARCH.1":
    st.header("RESEARCH.1 인공지능 기반 적정 카페밀도 예측")
    st.markdown('-----')
    RESEARCH1_menu = ["1. 연구배경", "2. 데이터 전처리", "3. 머신러닝 모델 학습","4. 딥러닝 모델 학습", "5. 시사점 및 제언"]
    RESEARCH1_choice = st.sidebar.selectbox("RESEARCH.1 메뉴", RESEARCH1_menu)

    if RESEARCH1_choice == "1. 연구배경":
        st.subheader("1. 연구배경")
        
        # 1.1 프로젝트 배경
        st.markdown('''
        #### 1.1 프로젝트 배경
        - 박소현 외 2명. "커피전문점 생존 및 폐업 분포의 군집 유형별 생멸 특성." 한국경제지리학회지 23.4 (2020): 408-424.
        "지난 10년(2010∼2019) 동안 서울지역에서 영업 중인 커피전문점의 수는 계속해서 증가했고, 증가분 폭은 다소 완만해졌지만, 폐업(률)보다 개업(률)이 더 많아 증가 추세를 유지하였다."<br/>
        - 도시 곳곳에서 급격하게 증가하는 카페들로 인해 과당경쟁이 심화되고 있습니다.
        - 이로 인한 문제를 해결하고 지속가능한 소상공인 생태계를 조성하기 위해 적정 카페 밀도의 예측이 필요합니다.
        - 카페 밀도의 최적화는 지역 경제에 긍정적인 영향을 미치고, 카페 운영자들에게 안정적인 사업 환경을 제공할 수 있습니다.
        ''',unsafe_allow_html=True)

        # 1.2 문제 정의
        st.markdown('''
        #### 1.2 문제 정의
        - 적정 카페 밀도는 신규 카페의 생존 가능성과 지속 가능한 매출을 동시에 고려한 수준을 의미합니다.
        - 이를 통해 카페 사업자들이 지속 가능한 성장을 이루고, 고객들에게 다양한 선택을 제공할 수 있습니다.
        ''',unsafe_allow_html=True)

        # 1.3 연구 방향성
        st.markdown('''
        #### 1.3 연구 방향성
        - 본 연구는 단위구역별 데이터를 활용하여 인공지능 모델을 통해 카페의 적정 밀도를 예측하고자 합니다.
        
        - 박소현 외 2명. "커피전문점 생존 및 폐업 분포의 군집 유형별 생멸 특성." 한국경제지리학회지 23.4 (2020): 408-424.
        "영업 중인 커피전문점의 밀집도가 높은 종로구와 강남구, 마포구 등에서 커피전문점 폐업도 높은 밀집도를 나타냈다"
        "본 연구 결과를 토대로 개⋅폐업 우세(열세)지역을 중심으로 입지 특성을 분석하거나 생존율(영업 대비 폐업비) 등 영업과 폐업을 포괄하는 공통의 지표를 마련하는 후속 연구를 통해 더 면밀한 입지분석 결과와 공간적 함의를 제시할 수 있을 것으로 기대한다."<br/>
        
        - 오세준, 김동억, 손병희.(2016).중소기업 프랜차이즈 점포의 매출 결정요인에 관한 연구.한국경영학회 융합학술대회,(),2168-2179.
        “입지 특성에서는 매장 주변의 유동 인구가 많을수록 매출이 증가되는 것으로 나타났다.”
        “점포 특성에서는 매장의 면적이 증가할수록 단위 면적 당 매출은 감소하는 것으로 나타났다.”<br/>
        
        - 민철기, 강창덕.(2021).상권의 공간적 확산에 따른 상업시설 생존율과 생존요인 비교 - 홍대지역 음식점을 중심으로 -.서울도시연구,22(2),17-38.
        “기존 연구에서 상업 시설의 매출에 영향을 미치는 주요 요인은 점포 특성, 교통 접근성, 경쟁, 인구 등으로 요약할 수 있다.”<br/>
        
        - 위의 선행 연구 등을 참고하여 카페 위치 데이터, 인구 통계, 경제 지표 등 다양한 데이터를 분석할 예정입니다.
        - 예측된 적정 밀도는 카페 사업자 및 정책 입안자에게 중요한 의사결정 정보를 제공할 것입니다.
        ''',unsafe_allow_html=True)

    elif RESEARCH1_choice == "2. 데이터 전처리":
        st.subheader("2. 데이터 전처리")
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
    | 17   | 17   | TBSM_SUD_WRC_POPLTN_QU | 표준단위구역집계_직장인구_분기   |"""
        
        st.markdown(" #### 2.1 데이터 목록")               
        st.markdown(markdown_table)
        st.markdown("")
        st.markdown('카페의 생존가능성과 매출에 관련된 변수와 표준단위구역 코드를 기준으로, 다른 데이터들과의 상관관계를 검증하여 트레이닝 데이터셋을 구축하고자 했다.')
        st.markdown("")
        image1_1 = Image.open("image/research1_heatmap2.png")
        st.image(image1_1, caption='Corr HeatMap')
        image1_2 = Image.open("image/research1_heatmap1.png")
        st.image(image1_2, caption='Corr HeatMap')
        st.markdown("")
        image1_3 = Image.open("image/research1_beingco.png")
        st.image(image1_3, caption='YEAR 3 BEING CO')
        st.markdown('단위구역의 작은 면적의 한계로 인해서 대부분의 3년 생존 기업수의 데이터의 분포가 0과 1에 몰려 있는 것을 발견 할 수 있었다.')
        image1_4 = Image.open("image/표준단위_상권.jpg")
        st.image(image1_4, caption='Scale Up')
        st.markdown("GIS를 활용해 기존 데이터보다 더 큰 상권 면적의 데이터를 재생산 할 수 있었다.")


    
    elif RESEARCH1_choice == "3. 머신러닝 모델 학습":
        st.subheader("3.  모델 학습")
        st.code('''features = ['YEAR_3_BEING_CO', 'YEAR_3_OPBIZ_STOR_CO', 'RELM_AR',
       'THSMON_SELNG_AMT',  'SIMILR_INDUTY_STOR_CO',
       'VIATR_FCLTY_CO',  'TOT_BSN_MT', 'TOT_STOR_CO',
        'OPBIZ_STOR_CO', 'CLSBIZ_STOR_CO',
       'FRC_STOR_CO']

x_input = r1_df[features]
cor_table = x_input.corr()

print(cor_table[abs(cor_table) > 0.9])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.set_title('Hierarchical dendrogram')
corr = spearmanr(x_input).correlation
corr_linkage = hierarchy.ward(corr)
dendro = hierarchy.dendrogram(corr_linkage, labels=features, ax=ax1, leaf_rotation=90)

dendro_idx = np.arange(0, len(dendro['ivl']))

ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
ax2.set_title('Feature correlation')
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
ax2.set_yticklabels(dendro['ivl'])

fig.tight_layout()''',language = 'python')
        image1_5 = Image.open("image/research1_traindataV.png")
        st.image(image1_5, caption='TrainData')
        st.markdown("")
        st.markdown('''
다중공선성과 상관관계를 고려하여 최종적으로 8개의 Feature와 1개의 Target을 설정하였다.
                    ''')
       
        st.markdown('''|    Column                 | Non-Null Count  | Dtype    |
|--------------------------|------------------|----------|
| 0   RELM_AR                | 3397 non-null   | float64  |
| 1   THSMON_SELNG_AMT       | 3397 non-null   | int64    |
| 2   YEAR_3_BEING_RT        | 3397 non-null   | float64  |
| 3   SIMILR_INDUTY_STOR_CO  | 3397 non-null   | int64    |
| 4   VIATR_FCLTY_CO         | 3397 non-null   | int64    |
| 5   TOT_BSN_MT             | 3397 non-null   | int64    |
| 6   TOT_STOR_CO            | 3397 non-null   | int64    |
| 7   OPBIZ_STOR_CO          | 3397 non-null   | int64    |
| 8   CLSBIZ_STOR_CO         | 3397 non-null   | int64    |
| 9   FRC_STOR_CO            | 3397 non-null   | int64    |''')
        
        st.code('''y_output = r1_df['THSMON_SELNG_AMT']

x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, random_state=42)
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model_name = []
train_perf = []
test_perf = []
                ''',language='python')
        st.markdown("""#### 3.1 엘라스틱 넷
- **THSMON_SELNG_AMT**를 타겟으로 하여 매출을 예측하는 회귀 머신러닝 모델을 제작.
릿지-랏소 회귀를 결합한 엘라스틱 넷을 활용하여 과적합 확률을 줄이고, GridSearch를 통해 최적의 하이퍼 파라미터를 찾아 학습을 진행""")
        st.code('''target = 'SIMILR_INDUTY_STOR_CO'
x_input=r1_df.drop(target,axis=1)
y_output = r1_df[target]

x_train, x_test, y_train, y_test = train_test_split(x_input, y_output, random_state=42)
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
model_name = []
train_perf = []
test_perf = []
                
en_model = ElasticNet(max_iter=100000000, random_state=42)

en_param = {'l1_ratio': np.arange(0.9, 0.995, 0.005),
            'alpha': np.arange(0.0008, 0.0028, 0.0001),
            'tol': [1e-2, 3.16e-2, 1e-1, 3.16e-1, 1e0]}
            

clf = GridSearchCV(estimator=en_model, param_grid=en_param, n_jobs=-1, verbose=3)
clf.fit(x_train_scaled, y_train)
y_train_pred = clf.predict(x_train_scaled)
y_test_pred = clf.predict(x_test_scaled)

print('Best_estimator:', clf.best_estimator_)
print('Best_params:', clf.best_params_)
print('Best_score:', clf.best_score_)
train_score = r2_score(y_train_pred, y_train)
print('Train score:', train_score)
test_score = r2_score(y_test_pred, y_test)
print('Test  score:', test_score)

model_name.append('Elastic Net')
train_perf.append(train_score)
test_perf.append(test_score)

plt.title('Prediction vs ground truth scatter plot (Elastic Net)')
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.grid()
plt.scatter(y_test_pred, y_test)
''',language='python')
        st.markdown('''|       | 점수                  |
|-------|-----------------------|
| Best  | 0.798228056241779     |
| Train | 0.8029349882637264    |
| Test  | 0.6622374048552624   |
''')
        image1_6 = Image.open("image/research_ML_1.png")
        st.image(image1_6, caption='Elastic Net')

        st.markdown("""#### 3.2 랜덤 포레스트""")
        
        st.code('''rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)

rf_param = {'n_estimators': range(76, 136, 20),
            'max_depth': range(30, 42, 4),
            'min_samples_split': range(2, 5, 1)}

clf = GridSearchCV(estimator=rf_model, param_grid=rf_param, n_jobs=-1, verbose=2)

clf.fit(x_train_scaled, y_train)
y_train_pred = clf.predict(x_train_scaled)
y_test_pred = clf.predict(x_test_scaled)

print('Best_estimator:', clf.best_estimator_)
print('Best_params:', clf.best_params_)
print('Best_score:', clf.best_score_)
train_score = r2_score(y_train_pred, y_train)
print('Train score:', train_score)
test_score = r2_score(y_test_pred, y_test)
print('Test  score:', test_score)

model_name.append('Random Forest')
train_perf.append(train_score)
test_perf.append(test_score)

plt.title('Prediction vs ground truth scatter plot (Random Forest)')
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.grid()
plt.scatter(y_test_pred, y_test)''',language='python')
        st.markdown('''|       | 점수                  |
|-------|-----------------------|
| Best  | 0.9368519928165371    |
| Train | 0.9930250814497149   |
| Test  | 0.9161962891032838    |
''')
    
        image1_7 = Image.open("image/research_ML_2.png")
        st.image(image1_7, caption='Random Forest')

        st.markdown('''''')
        st.code('''rf_model.fit(x_train_scaled, y_train)

feature_names = ['feature {i}' for i in range(x_train_scaled.shape[1])]

importances = rf_model.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_model.estimators_], axis=0)

forest_importances = pd.Series(importances, index=features)
forest_importances = forest_importances.sort_values(ascending=False)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()''',language='python')
        image1_8 = Image.open("image/research_ML_3.png")
        st.image(image1_8, caption='Feature importances')
        
        st.markdown('''''')
        st.code('''result = permutation_importance(rf_model, x_train_scaled, y_train, n_repeats=100, n_jobs=-1, random_state=42)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=x_train.columns[sorted_idx])
ax.set_title("Permutation importances (train set)")
fig.tight_layout()''',language='python')
        image1_9 = Image.open("image/research_ML_4.png")
        st.image(image1_9, caption='Permutation importance')
        st.markdown("""#### 3.3 XGBRegressor (GridSearch)""")

        st.code('''xgb_model = XGBRegressor(n_jobs=-1, random_state=42)

xgb_param = {'n_estimators': range(80, 140, 20),
             'max_depth': range(2, 5, 1),
             'reg_alpha': [1e-2, 3.16e-2, 1e-1],
             'reg_lambda': [0.1, 0.316, 1]}

xgb_clf = GridSearchCV(estimator=xgb_model, param_grid=xgb_param, n_jobs=-1, verbose=2, cv=3)

xgb_clf.fit(x_train_scaled, y_train)
y_train_pred = xgb_clf.predict(x_train_scaled)
y_test_pred = xgb_clf.predict(x_test_scaled)

print('Best_estimator:', xgb_clf.best_estimator_)
print('Best_params:', xgb_clf.best_params_)
print('Best_score:', xgb_clf.best_score_)
train_score = r2_score(y_train_pred, y_train)
print('Train score:', train_score)
test_score = r2_score(y_test_pred, y_test)
print('Test  score:', test_score)

model_name.append('XGBoost')
train_perf.append(train_score)
test_perf.append(test_score)

plt.title('Prediction vs ground truth scatter plot (XGBoost)')
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.grid()
plt.scatter(y_test_pred, y_test)''',language='python')
        st.markdown('''|       | 점수                  |
|-------|-----------------------|
| Best  | 0.9390181469001927    |
| Train | 0.9846099860284766    |
| Test  | 0.908798759050328    |
''')
        image1_10 = Image.open("image/research_ML_5.png")
        st.image(image1_10, caption='XGBoost')

        st.markdown("""#### 3.4 KNN""")
        st.code('''svr_model = SVR(C=1e8, gamma=0.1)
knn_model = KNeighborsRegressor(n_jobs=-1)

knn_param = {'n_neighbors': range(1, 6, 1),
             'weights': ['uniform', 'distance'],
             'leaf_size': range(1, 5, 1)}

knn_clf = GridSearchCV(estimator=knn_model, param_grid=knn_param, n_jobs=-1, verbose=3)

knn_clf.fit(x_train_scaled, y_train)
y_train_pred = knn_clf.predict(x_train_scaled)
y_test_pred = knn_clf.predict(x_test_scaled)

print('Best_estimator:', knn_clf.best_estimator_)
print('Best_params:', knn_clf.best_params_)
print('Best_score:', knn_clf.best_score_)
print('Train score:', r2_score(y_train_pred, y_train))
print('Test  score:', r2_score(y_test_pred, y_test))

plt.title('Prediction vs ground truth scatter plot (kNN)')
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.grid()
plt.scatter(y_test_pred, y_test)''',language='python')
        st.markdown('''|       | 점수                  |
|-------|-----------------------|
| Best  | 0.9094165069559672    |
| Train | 1.0    |
| Test  | 0.8282274874565931  |
''')
        image1_11 = Image.open("image/research_ML_6.png")
        st.image(image1_11, caption='KNN')

        st.markdown("""#### 3.4 KNN""")
        st.code('''svr_model = SVR(C=1e8, gamma=0.1)
knn_model = KNeighborsRegressor(n_jobs=-1)

knn_param = {'n_neighbors': range(1, 6, 1),
             'weights': ['uniform', 'distance'],
             'leaf_size': range(1, 5, 1)}

knn_clf = GridSearchCV(estimator=knn_model, param_grid=knn_param, n_jobs=-1, verbose=3)

knn_clf.fit(x_train_scaled, y_train)
y_train_pred = knn_clf.predict(x_train_scaled)
y_test_pred = knn_clf.predict(x_test_scaled)

print('Best_estimator:', knn_clf.best_estimator_)
print('Best_params:', knn_clf.best_params_)
print('Best_score:', knn_clf.best_score_)
print('Train score:', r2_score(y_train_pred, y_train))
print('Test  score:', r2_score(y_test_pred, y_test))

plt.title('Prediction vs ground truth scatter plot (kNN)')
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.grid()
plt.scatter(y_test_pred, y_test)''',language='python')
        st.markdown('''|       | 점수                  |
|-------|-----------------------|
| Best  | 0.9094165069559672    |
| Train | 1.0    |
| Test  | 0.8282274874565931  |
''')
        image1_11 = Image.open("image/research_ML_6.png")
        st.image(image1_11, caption='KNN')



        st.markdown("""#### 3.5 MLP""")
        st.code('''mlp_model = MLPRegressor(learning_rate='adaptive', max_iter=2000000, verbose=0,
                         early_stopping=True, random_state=42)

mlp_param = {'hidden_layer_sizes': [x for x in itertools.product((range(24, 32)), repeat=4)],
             'alpha': loguniform.stats(1e-4, 1e-3)}

rand_clf = RandomizedSearchCV(estimator=mlp_model, param_distributions=mlp_param, n_iter=10, n_jobs=-1,
                              cv=3, verbose=2, random_state=42)

rand_clf.fit(x_train_scaled, y_train)

y_train_pred = rand_clf.predict(x_train_scaled)
y_test_pred = rand_clf.predict(x_test_scaled)

print('Best_estimator:', rand_clf.best_estimator_)
print('Best_params:', rand_clf.best_params_)
print('Best_score:', rand_clf.best_score_)
train_score = r2_score(y_train_pred, y_train)
print('Train score:', train_score)
test_score = r2_score(y_test_pred, y_test)
print('Test  score:', test_score)

model_name.append('Neural Network')
train_perf.append(train_score)
test_perf.append(test_score)
plt.title('Prediction vs ground truth scatter plot (Neural Network)')
plt.xlabel('Prediction')
plt.ylabel('Ground truth')
plt.grid()
plt.scatter(y_test_pred, y_test)
''',language='python')
        st.markdown('''|       | 점수                  |
|-------|-----------------------|
| Best  | 0.8640667781720607    |
| Train | 0.8490975601150919    |
| Test  | 0.8555902522570816  |
''')
       
        image1_11 = Image.open("image/MLP.png")
        st.image(image1_11, caption='MLP')





        st.markdown("""#### Model performance comparison""")
        image1_12 = Image.open("image/research_ML_7.png")
        st.image(image1_12, caption='Model performance comparison')
        st.markdown("""#### Ensemble""")
        st.code('''ensemble_list = [('xgb', xgb_model),
                 ('rf', rf_model),
                 ('svm', svr_model),
                 ('en', en_model)]

voting_est = []

for idx, target_model in enumerate(ensemble_list):
    voting_est.append(target_model)

    voting_clf = VotingRegressor(estimators=voting_est, n_jobs=-1)
    voting_clf.fit(x_train_scaled, y_train)
    y_train_pred = voting_clf.predict(x_train_scaled)
    y_test_pred = voting_clf.predict(x_test_scaled)
    print(idx + 1, 'Ensemble train score:', r2_score(y_train_pred, y_train))
    print(idx + 1, 'Ensemble test  score:', r2_score(y_test_pred, y_test))

    plt.scatter(y_test_pred, y_test, label=str(idx + 1), alpha=0.5)
    title = 'Prediction vs ground truth scatter plot: Ensemble'     
    plt.title(title)
    plt.xlabel('Prediction')
    plt.ylabel('Ground truth')
    plt.grid()
    plt.legend()''',language='python')
        st.markdown('''
| 앙상블 번호 | 훈련 점수          | 테스트 점수         |
|------------|------------------|------------------|
| 1          | 0.9813709793460065 | 0.9532599775921411 |
| 2          | 0.9884039624194397 | 0.9603322554899544 |
| 3          | 0.994592381664609  | 0.9580324860380139 |
| 4          | 0.981559176737165  | 0.9491235130078842 |
| 5          | 0.9621853402127016 | 0.9339185178037513 |

''')
        image55_1 = Image.open("image/ensemble_score.png")
        st.image(image55_1, caption='Ensemble_score')
        image1_13 = Image.open("image/research_ML_8.png")
        st.image(image1_13, caption='Ensemble')
        st.markdown("""#### 3.7 VotingRegressor""")
        st.code('''voting_clf = VotingRegressor(estimators=[('xgb', xgb_model), ('rf', rf_model), ('svm', svr_model)],
                             n_jobs=-1)
voting_clf.fit(x_train_scaled, y_train)
                result = permutation_importance(voting_clf, x_train_scaled, y_train, n_repeats=100, n_jobs=-1, random_state=42)
sorted_idx = result.importances_mean.argsort()

fig, ax = plt.subplots()
ax.boxplot(result.importances[sorted_idx].T, vert=False, labels=x_train.columns[sorted_idx])
ax.set_title("Permutation importances (train set)")
fig.tight_layout()''',language='python')
        
        image1_14 = Image.open("image/research_ML_9.png")
        st.image(image1_14, caption='Permutation importance')

        st.markdown("""#### 결과 예측""")
        image1_15 = Image.open("image/KakaoTalk_20240129_133734927_04.png")
        st.image(image1_15, caption='결과 예측')
        st.markdown('''''')

        



        
        

    
    elif RESEARCH1_choice == "4. 딥러닝 모델 학습":
        
        st.subheader("4. 딥러닝 모델 학습")
        st.markdown('''처음 모델의 방향으로는 일반적인 MLP를 가지고 다변량 회귀 모델로 성능을 높이는데 주력하였다. 모델에서 활성화 함수로 최근에 가장 많이 사용되는 
                    GELU 함수를 사용하였고 과적합 방지를 위해 Dropout을 사용했다. 하지만 학습 결과는 그닥 좋지 않았다.''')
        
        st.code('''print('MLP')

print(metrics.r2_score(y_test, pred))''', language = 'python')
        st.markdown('''MLP  
0.24227769126612708''')
        st.markdown('''그래서 모델 설계를 새로 하게 되었다. 기존의 ResNet에서 사용되었던 Skip Connection 구조를 활용함으로써 
                    더 깊은 신경망 구조에서 기울기 소실 및 폭주가 일어나지 않고 잘 학습할 수 있도록 하였다.''')
        
        st.code('''class SkipMLP(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim[0])
        self.ln2 = nn.LayerNorm(hidden_dim[1])
        self.ln3 = nn.LayerNorm(hidden_dim[2])
        self.ln4 = nn.LayerNorm(hidden_dim[3])
        
        self.up1 = nn.Sequential(nn.Linear(input_dim, hidden_dim[0]),
                                 nn.GELU(),
                                 nn.BatchNorm1d(hidden_dim[0]))
        
        self.up2 = nn.Sequential(nn.Linear(hidden_dim[0], hidden_dim[1]),
                                 nn.GELU(),
                                 nn.BatchNorm1d(hidden_dim[1]))
        
        self.up3 = nn.Sequential(nn.Linear(hidden_dim[1], hidden_dim[2]),
                                 nn.GELU(),
                                 nn.BatchNorm1d(hidden_dim[2]))
        
        self.up4 = nn.Sequential(nn.Linear(hidden_dim[2], hidden_dim[3]),
                                 nn.GELU(),
                                 nn.BatchNorm1d(hidden_dim[3]))
        
        self.down1 = nn.Sequential(nn.Linear(hidden_dim[3], hidden_dim[2]),
                                   nn.GELU(),
                                   nn.BatchNorm1d(hidden_dim[2]))
        
        self.down2 = nn.Sequential(nn.Linear(hidden_dim[2], hidden_dim[1]),
                                   nn.GELU(),
                                   nn.BatchNorm1d(hidden_dim[1]))
        
        self.down3 = nn.Sequential(nn.Linear(hidden_dim[1], hidden_dim[0]),
                                   nn.GELU(),
                                   nn.BatchNorm1d(hidden_dim[0]))
        
        self.down4 = nn.Sequential(nn.Linear(hidden_dim[0], 64),
                                   nn.GELU(),
                                   nn.BatchNorm1d(64))
        
        self.fc = nn.Linear(64, 1)
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        up1 = self.up1(x)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        up4 = self.up4(up3)
        
        down1 = self.down1(self.ln4(up4))
        skip1 = down1 + up3
        down2 = self.down2(self.ln3(skip1))
        skip2 = down2 + up2
        down3 = self.down3(self.ln2(skip2))
        skip3 = down3 + up1
        down4 = self.down4(self.ln1(skip3))
        
        out = self.fc(down4)
        
        return out''', language = 'python')
        st.markdown('''위 모델로 학습했을 때의 결과는 다음과 같다.''')
        st.code('''print('SkipMLP')
print(metrics.r2_score(pred, y_test))''', language = 'python')
        st.markdown('''SkipMLP  
0.22331372022970997''')
        
        st.markdown('''최근 딥러닝 모델에서의 추세는 거대한 모델과 엄청난 양의 데이터셋으로 pretraining 후에 원하는 데이터 셋을 fine-tuning하는 방식으로 
                    성능을 높이는데 그러한 방식을 여기에도 적용해보았다. Pretraining 시에는 모든 업종의 데이터를 활용해 학습을 시키고 fine-tuning
                    시에는 카페 업종의 데이터만을 활용해 학습을 시켜 성능을 높일 수 있었다.
                    <br/>
                    훈련시킬 때 pretraining 때에 optimizer는 AdamW, fine-tuning 때에는 Adam을 사용하였고 pretraining, fine-tuning 모두 
                    학습률 스케쥴러로 CosineAnnealingLR을 사용했다.''')
        
        st.code('''pre_model = SkipMLP(input_dim = 10, hidden_dim = [128, 256, 512, 1024])
pre_model.apply(init_weights).to(device)

criterion = nn.MSELoss()
pre_optimizer = optim.AdamW(params = pre_model.parameters(), lr = cfg['adamw_lr'], weight_decay = cfg['weight_decay'])
pre_scheduler = lr_scheduler.CosineAnnealingLR(optimizer = pre_optimizer, T_max = 20, eta_min = 1e-8)''', language = 'python')
        st.code('''tune_model = SkipMLP(input_dim = 10, hidden_dim = [128, 256, 512, 1024]).to(device)

pre_path = 'Pretraining_weight'
ckpt = torch.load(pre_path)
tune_model.load_state_dict(ckpt)

criterion = nn.MSELoss()
tune_optimizer = optim.Adam(params = tune_model.parameters(), lr = cfg['adam_lr'])
tune_scheduler = lr_scheduler.CosineAnnealingLR(optimizer = tune_optimizer, T_max = 20, eta_min = 1e-8)''', language = 'python')
        st.markdown('''다음과 같은 모델과 학습 방법으로 학습시킨 결과는 다음과 같다.''')
        st.code('''print('pre-train r2: {}'.format(metrics.r2_score(y_test, pretrain_pred)))
print('fine-tuning r2: {}'.format(metrics.r2_score(fine_y_test, tuning_pred)))''', language = 'python')
        st.markdown('''pre-train r2: 0.69575578989722  
fine-tuning r2: 0.6623934253064305''')
        st.markdown('''일반적인 MLP로도 pre-training과 fine-tuning을 수행해봤을 때 결과는 다음과 같았다.''')
        st.code('''print('pre-train r2: {}'.format(metrics.r2_score(y_test, pretrain_pred)))
print('fine-tuning r2: {}'.format(metrics.r2_score(fine_y_test, tuning_pred)))''', language = 'python')
        st.markdown('''pre-train r2: 0.4717688378688347  
fine-tuning r2: 0.24227769126612708''')
        image1_12 = Image.open('image/output.png')
        st.image(image1_12)
        image1_13 = Image.open('image/output2.png')
        st.image(image1_13)
        st.markdown('''위 사진은 각 데이터의 Test set에서의 추론 결과다.''')
        st.markdown('''카페 점포 수를 늘려가며 fine-tuning된 모델로 카페의 점포수를 예측해보았다.''')
        image1_14 = Image.open('image/output3.png')
        st.image(image1_14)
        

    elif RESEARCH1_choice == "5. 시사점 및 제언":
        st.subheader("5. 시사점 및 제언")
        st.markdown('''#### 5.1 시사점
- 딥러닝에서는 기본적인 MLP 모델에서 skip connection을 통해 더 깊이 쌓을 수 있게 되며 더 좋은 결과를 초래할 수 있었고, pre-training, fine-tuning을 사용해 
                    카페 데이터만을 학습했을 때보다 더 좋은 결과를 도출했다. r2score가 약 0.7로 주어진 데이터의 상당 부분을 잘 설명하도록 모델을 학습을 하였음
- 적정 카페 밀도를 예측함으로서 카페 사업자들에게 지속 가능한 사업 운영을 위한 중요한 기준을 제공가능할 것으로 예상 된다.                   
                    ''')
        st.markdown('''#### 5.2 제언
- 모델을 활용해서, 유전알고리즘을 적용하면 최대의 점포매출을 산출하는 점포 수를 도출해 낼 것이라 생각됨.                   
                    ''')
 
        image1_16 = Image.open("image/pso_shcb.gif")
        st.image(image1_16, caption='유전 알고리즘')

    

elif choice == "RESEARCH.2":
    st.header("RESEARCH.2 시계열 분석 기반 미래 카페 수 예측")
    st.markdown('-----')
    RESEARCH2_menu = ["1. 연구배경", "2. 데이터 전처리", "3. 모델 학습", "4. 결과 확인", "5. 시사점 및 제언"]
    RESEARCH2_choice = st.sidebar.selectbox("RESEARCH.2 메뉴", RESEARCH2_menu)
    if RESEARCH2_choice == "1. 연구배경":
        st.subheader("1. 연구배경")
        st.markdown("- **1.1 프로젝트 배경**")
        st.markdown('''
        - 지역 주민의 일상생활과 밀접한 재화와 서비스를 제공함으로써 서민경제의 근간을 이루고 있으며, 점포 개설이 상대적으로 쉬운 도·소매업, 음식업 등의 업종
                    은 주로 진입장벽이 낮은 골목상권에 주로 진입하는 양상을 보이고 있습니다.
        - 프렌차이즈 카페 산업의 급속한 성장과 더불어 프렌차이즈 카페 뿐 아니라, 상대적으로 작은 규모로 창업이 가능한 개인 카페의 숫자가 급증하고 있습니다.
        - 단위구역에서 가까운 미래의 카페 수를 예측하여 소상공인에게 유용한 정보를 제공하고, 창업계획수립에 기여하고자 합니다.
        ''')
        st.markdown("- **1.2 문제 정의**")
        st.markdown('''
        - 본 연구는 분기별 시계열 데이터와 카페의 총 점포 수를 분석하여, 미래의 카페 수를 예측하는 시계열 예측 모델을 개발하는 것을 목표로 합니다.
        - 이를 통해 카페 사업자 및 정책 입안자들에게 실질적인 도움을 제공할 수 있습니다.
        ''')
        image3_1 = Image.open("image/research2_1.webp")
        st.image(image3_1, caption='시계열 예측모델 예시')
    elif RESEARCH2_choice == "2. 데이터 전처리":
        st.subheader("2. 데이터 전처리")
        st.markdown("- **2.1 데이터 분석**")
        st.markdown('''
        - 미래 카페 점포 수 예측을 위해, 이전 연구에서 분석한 데이터 간의 상관관계를 활용합니다.
        - 카페 총점포수와 관련된 다양한 피쳐들을 결합하여 예측 모델의 정확도를 높이고자 합니다.
        ''')
        image3_2 = Image.open("image/research2_1.png")
        st.image(image3_2, caption='데이터 시각화')
        st.markdown("- **2.2 데이터 전처리 과정**")
        st.markdown('''
        - 기준 년도와 분기를 나타내는 'STDR_YYQU_CD' 데이터를 시계열 분석에 적합하게 전처리합니다.
        - 2019년부터 2023년까지의 분기별 데이터만으로는 시계열 분석에 한계가 있으므로, 시계열 데이터를 보간기법을 활용하여 시계열 데이터를 추가합니다.
        - 단위구역별로 더 세밀한 시계열 데이터를 통해, 예측 모델의 정확도를 높일 수 있는 기반을 마련합니다.
        ''')
        image3_3 = Image.open("image/research2_2 (2).png")
        st.image(image3_3, caption='시계열 데이터 보간')
        st.markdown('''
1. **데이터 로드**: 'pd.read_csv'를 사용하여 데이터를 로드합니다.
2. **행 수 기반 SUD_CD 제거**: 각 'SUD_CD'별로 '날짜' 열의 행 수를 확인하여 10개 이하인 경우 제거합니다.
3. **날짜를 datetime으로 변환**: '날짜' 열을 datetime 타입으로 변환합니다.
4. **보간할 열과 보존할 열 지정**: 보간할 열과 그 외의 열을 지정합니다.
5. **각 SUD_CD별로 처리**: 각 'SUD_CD'에 대해 별도로 데이터 처리를 진행합니다.
6. **중복된 날짜 제거 및 월별 데이터로 resample**: 중복된 날짜를 제거하고 월별 데이터로 재표본화합니다.
7. **지정된 열 보간 및 나머지 열 채우기**: 지정된 열에 대해 보간을 수행하고, 나머지 열은 가까운 시간대 데이터로 채웁니다.
8. **SUD_CD 열 추가 및 리스트에 저장**: 'SUD_CD' 열을 추가하고 결과를 리스트에 저장합니다.
9. **모든 resampled 데이터를 하나의 DataFrame으로 결합**: 처리된 모든 데이터를 하나의 DataFrame으로 결합합니다.
''')
        
    
    

    elif RESEARCH2_choice == "3. 모델 학습":
        st.subheader("3. 모델 학습")
        st.markdown("- **3.1 학습 데이터셋 검증**")
        st.markdown('''
        - 본 연구에서는 K-Fold 교차 검증 방식에서 착안하여, 각 단위구역별로 데이터셋을 분리하여 모델 학습을 진행합니다.
        - 이를 통해 각 지역의 특성을 반영한 보다 정확한 시계열 예측이 가능합니다.
        ''')
        image3_4 = Image.open("image/research_heatmap.png")
        st.image(image3_4, caption='학습 데이터셋 검증')
        st.markdown("- **3.2 시계열 예측 모델 적용**")
        st.markdown('''
        - 전체 156개 단위구역 중에서 데이터가 부족한 23개 구역을 제외하고, 남은 133개 구역에 대해 딥러닝 모델을 개별적으로 구축합니다.
        - 이 133개의 모델을 앙상블 방식으로 결합하여, 더욱 정확하고 신뢰성 있는 시계열 예측을 목표로 합니다.
        - 이러한 접근은 일반적인 시계열 예측보다 효율적이고 정밀한 결과를 도출할 것으로 기대됩니다.
        ''')
        image3_5 = Image.open("image/LSTM_1.png")
        st.image(image3_5, caption='LSTM')
        image3_6 = Image.open("image/K-FOLD.png")
        st.image(image3_6, caption='K-FOLD')

    elif RESEARCH2_choice == "4. 결과 확인":
        st.subheader("4. 결과 확인")
        st.markdown("- 4.1 학습 결과 확인")
        image3_7 = Image.open("image/Research2_training.png")
        st.image(image3_7, caption='Training')
        st.markdown('''
        - 각 단위구역에 대해서 개별적인 시계열 학습
        - 단위구역의 총점포수의 변화를 타겟으로하여, 시계열에 따라 변화하는 점포수 예측 모델을 생성합니다.
        - 추후, 모델 향상을 위한 앙상블 학습을 위해 각 단위구역마다 모델을 생성합니다.
                    ''')
        st.markdown("- 4.2 예측 결과 확인")
        st.markdown('''
        - 각 단위구역 별 학습 모델의 예측과 실제 데이터를 비교하여 적절히 학습되었는지 확인합니다.
                    ''')
        image3_8 = Image.open("image/research2_4.png")
        st.image(image3_8, caption='Validation')

    elif RESEARCH2_choice == "5. 시사점 및 제언":
        st.subheader("5. 시사점 및 제언")
        st.markdown("- 5.1 앙상블 기법 제안")
        st.markdown('''
        - 단위구역 별 학습한 133개의 모델을 앙상블 학습을 통해 하나의 일반적인 모델을 구축하여 그 지역의 데이터가 있다면 미래의 점포 수를 예측할 수 있을 것으로 예상합니다.
                    ''')
        st.markdown("- 5.2 시사점 및 제언")
        st.markdown('''
        - 시간의 흐름에 따른 점포수의 변동을 기반으로 미래의 점포수 예측을 Research 1, Research 2와 결합하여
                    더 높은 정확도의 예측을 구현할 수 있습니다.
        - 서울시 상권분석 서비스와의 연계를 바탕으로 상권의 발전 양상을 파악할 수 있습니다. 
                    ''')


elif choice == "RESEARCH.3":
    st.header("RESEARCH.3 카페와 시너지, 링겔만 효과를 보여주는 업종 판별")
    st.markdown('-----')
    RESEARCH3_menu = ["1. 연구배경", "2. 데이터 전처리", "3. 모델 학습", "4. 결과 확인", "5. 시사점 및 제언"]
    RESEARCH3_choice = st.sidebar.selectbox("RESEARCH.3 메뉴", RESEARCH3_menu)

    if RESEARCH3_choice == "1. 연구배경":
    # 1.1 프로젝트 배경
        st.subheader("1.1 프로젝트 배경")
        st.markdown('''
        **카페 산업의 변화와 도전**
        
        - 최근 몇 년 간 카페의 집적 밀도가 높아지고, 과당경쟁으로 인한 문제가 발생하고 있습니다. 
        - 이러한 상황에서 카페와 상생할 수 있는 다른 업종에 대한 연구는 지속 가능한 상업 환경을 
        조성하는 데 매우 중요합니다.
        ''')

        # 1.2 문제 정의
        st.subheader("1.2 문제 정의")
        st.markdown('''
        **상생의 정의**

        - 이 연구에서는 같은 시간대에 카페를 방문한 후 타 업종을 방문하는 경우, 혹은 그 반대의 상황일 때 해당 업종을 상생한다고 정의합니다.
        ''')

    elif RESEARCH3_choice == "2. 데이터 전처리":
        st.header("2. 데이터 전처리")
        st.markdown('-----')

        # 3.1 데이터 선정
        st.subheader("2.1 데이터 선정")
        st.markdown('''
        **핵심 데이터의 선택**
            
            - 노기호(Giho Roh) 외 2인. "아울렛 출점이 주변 점포에 미치는 영향." 流通硏究 24.1 (2019): p2. 

            "상권 전체에 미치는 영향을 분석하기 위해 대규모 점포와 대규모점포 주변 점포들의 카드 매출액 분석을 이용하였다."


            - 류준영 외 2인.(2014).서울지역 창업 중소기업의 생존율과 생존기간에 영향을 미치는 요인 연구.도시행정학보,27(4),250-251.

            "총 112개(부실기업 47개, 건전기업 65개)의 기업을 분석한 결과, 도산예측의 중요변수로 자기자본 순이익률, 매출액 총이익률, 차입금 의존도, 자산총액을 제시하였다."
            
            
        - 본 연구에서는 위의 논문 및 선행 연구를 참고하여, **매출액**을 핵심 데이터로 선택하였습니다.
        ''')

        # 3.2 데이터 전처리
        st.subheader("2.2 데이터 전처리")
        st.markdown('''
        **데이터 전처리 과정**
        - 1.2 문제 정의를 토대로 데이터는 **시간대별 매출**을 사용하였습니다.
        - 시간대별 매출 데이터는 업종 전체의 매출임을 감안하여 점포 수로 나눠 **점포 당 시간대별 매출 비율 데이터**로 사용하였습니다.
        - 타 업종과 카페와의 관계를 분석하기 위해 **카페가 존재하지 않는 표준단위구역의 데이터는 제거**하였습니다.
        - 점포 별로 영업시간이 상이하며 영업 시작을 하지 않았을 경우 매출이 0원으로 나타나므로 **매출이 0원인 데이터는 제거**하였습니다.
        ''')

        # 3.3 분석 방법 선정
        st.subheader("2.3 분석 방법 선정")
        st.markdown('''
        **군집 분석을 통한 상생 및 악생 관계 파악**

        - 카페와 비슷한 매출 양상을 가지는 업종을 식별하기 위해 군집 분석을 사용하였습니다.
        - 군집 분석을 통해 카페의 매출이 낮은 군집(악생)과 매출이 높은 군집(상생)을 구분합니다.
        - 이를 통해 각 군집의 비율에 따라 카페와 다른 업종 간의 상생 및 악생 관계의 정도를 평가합니다.
        ''')
       





        # Streamlit 코드 계속
    elif RESEARCH3_choice == "3. 모델 학습":
        st.header("3. 모델 학습")
        st.markdown('-----')
        
        st.subheader("3.1 모델 특성(Feature) 선정")
        st.markdown('''
        **모델 특성(Feature)의 중요성**

        - 군집 분석에서 효과적인 결과를 얻기 위해서는 올바른 feature(특성 또는 변수)를 선택하는 것이 중요합니다. Feature의 선정은 군집화 모델의 성능과 해석력에 큰 영향을 미칩니다.
        - 이를 위해 RFE 및 RFECV를 사용하였습니다.
        - RFE(Recursive Feature Elimination): 모델 성능을 유지하면서 중요하지 않은 특성을 반복적으로 제거하는 특성 선택 방법.
        - RFECV(Recursive Feature Elimination with Cross-Validation): RFE에 교차 검증을 추가하여 최적의 특성 개수를 자동으로 찾는 방법.

        ''')
        st.code('''from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import matplotlib.pyplot as plt

# 업종코드를 target으로 설정
target_col = "SVC_INDUTY_CD"
X = df.drop(columns=[target_col])
y = df[target_col]

# 분류 모델 선택 (RandomForestClassifier 사용)
model = RandomForestClassifier()

# RFECV 모델 생성
rfecv1 = RFECV(estimator=model, step=1, cv=5)  # cv는 교차 검증 폴드 수입니다.

# 모델을 훈련하고 최적의 변수 수를 선택
rfecv1 = rfecv1.fit(X, y)

# 선택된 변수의 개수 출력
print("Optimal number of features : %d" % rfecv1.n_features_)

# 선택된 변수의 인덱스 출력
print("Selected features indices : %s" % rfecv1.support_)

# 선택된 변수 시각화
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv1.cv_results_['mean_test_score']) + 1), rfecv1.cv_results_['mean_test_score'])
plt.show()
                ''',language='python')
        image2_1 = Image.open("image/RFECV_RandomForest.png")
        st.image(image2_1, caption='Random Forest 알고리즘 기반 RFECV')
        st.code('''from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()

target_col = "SVC_INDUTY_CD"
X = df.drop(columns=[target_col])
y = df[target_col]

# RFECV 모델 생성
rfecv4 = RFECV(estimator=model, step=1, cv=5)  # cv는 교차 검증 폴드 수입니다.

# 모델을 훈련하고 최적의 변수 수를 선택
rfecv4 = rfecv4.fit(X, y)

# 선택된 변수의 개수 출력
print("Optimal number of features : %d" % rfecv4.n_features_)

# 선택된 변수의 인덱스 출력
print("Selected features indices : %s" % rfecv4.support_)

# 선택된 변수 시각화
import matplotlib.pyplot as plt
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv4.cv_results_['mean_test_score']) + 1), rfecv4.cv_results_['mean_test_score'])
plt.show()
                ''',language='python')
        image2_2 = Image.open("image/RFECV_DecisionTree.png")
        st.image(image2_2, caption='Decision Tree 알고리즘 기반 RFECV')



        # 4.1 적정 Cluster 수 선정
        st.subheader("3.2 적정 클러스터(Cluster) 수 선정")
        st.markdown('''
        **클러스터 수의 중요성**

        - 클러스터 수가 너무 많거나 적으면 분석의 정확도와 유용성이 감소할 수 있습니다.
        - 이를 위해 실루엣 점수(Silhouette Score)를 사용하여 각 클러스터 간의 거리와 응집도를 평가하였습니다.

        ''')
            
        st.code('''from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 군집분석을 위한 Feature 선택
features = ['TMZON_06_11_SELNG_RATE', 'TMZON_11_14_SELNG_RATE',
            'TMZON_14_17_SELNG_RATE', 'TMZON_17_21_SELNG_RATE',
            'TMZON_21_24_SELNG_RATE']

# Target 설정
target = 'SVC_INDUTY_CD'

# Feature와 Target을 선택하여 새로운 데이터 프레임 생성
df_selected = df[[target] + features]

# 데이터 표준화
scaler = StandardScaler()
df_selected[features] = scaler.fit_transform(df_selected[features])

silhouette_scores = []
K = range(2, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(df_selected[features])
    score = silhouette_score(df_selected[features], kmeanModel.labels_)
    silhouette_scores.append(score)

plt.figure(figsize=(8, 4))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score showing the optimal k')
plt.show()
                ''',language='python')
        
        image2_4 = Image.open("image/SilhouetteScore.png")
        st.image(image2_4, caption='Silhouette Score에 따른 클러스터 수 결정')

        # 4.2 군집분석 진행
        st.subheader("3.3 군집분석 진행")
        st.markdown('''
        **군집분석 알고리즘 선택과 적용**
        
        조지환, 김동현, 김현, 김민석, 주종화.(2022).서울시 카페 입지선정 도움 서비스.한국정보기술학회 종합학술발표논문집,(),767-772.
        "군집방법으로는  계층적  군집방법인 Hierarchical  Clustering과  비계층적  군집방법인 K-means  clustering을 진행하였고 그 결과 모두 최적의 군집수는 4개였다.
        
        - 위 선행 연구를 토대로 본 연구에서는 두 가지 군집분석 알고리즘을 사용하였습니다: 계층적 군집분석(Agglomerative Clustering)과 비계층적 군집분석(K-Means).
        - 계층적 군집분석(Agglomerative Clustering)은 점차적으로 유사한 데이터 포인트를 결합하여 클러스터를 형성합니다.
        - K-Means 알고리즘은 데이터 포인트를 K개의 클러스터로 그룹화하여 각 클러스터의 중심점을 찾습니다.
        - 이러한 알고리즘을 통해 2개에서 5개까지 다양한 클러스터 수를 실험하며 최적의 구조를 찾아냈습니다.
        ''')
        st.code('''import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt


# 군집분석을 위한 Feature 선택
features = ['TMZON_06_11_SELNG_RATE', 'TMZON_11_14_SELNG_RATE',
            'TMZON_14_17_SELNG_RATE', 'TMZON_17_21_SELNG_RATE',
            'TMZON_21_24_SELNG_RATE']

# Target 설정
target = 'SVC_INDUTY_CD'

# Feature와 Target을 선택하여 새로운 데이터 프레임 생성
df_selected4 = df[[target] + features]

# Agglomerative Clustering 모델 생성
agglomerative = AgglomerativeClustering(n_clusters=2)  # 클러스터 개수는 적절하게 조절 가능
df_selected4['cluster'] = agglomerative.fit_predict(df_selected4[features])

# Pairplot 그리기
sns.set(style="ticks")
sns.pairplot(df_selected4, hue='cluster')
plt.show()
                ''',language='python')

        image2_5 = Image.open("image/AgglomerativeClustering_2.png")
        st.image(image2_5, caption='2개 클러스터로 진행한 Agglomerative Clustering')
        image2_6 = Image.open("image/AgglomerativeClustering_4.png")
        st.image(image2_6, caption='4개 클러스터로 진행한 Agglomerative Clustering')
        st.code('''import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 군집분석을 위한 Feature 선택
features = ['TMZON_06_11_SELNG_RATE', 'TMZON_11_14_SELNG_RATE',
            'TMZON_14_17_SELNG_RATE', 'TMZON_17_21_SELNG_RATE',
            'TMZON_21_24_SELNG_RATE']

# Target 설정
target = 'SVC_INDUTY_CD'

# Feature와 Target을 선택하여 새로운 데이터 프레임 생성
df_selected = df[[target] + features]

# 데이터 표준화
scaler = StandardScaler()
df_selected[features] = scaler.fit_transform(df_selected[features])

# k-means 군집 모델 생성
kmeans = KMeans(n_clusters=2, random_state=42)  # 클러스터 개수는 적절하게 조절 가능
df_selected['cluster'] = kmeans.fit_predict(df_selected[features])

# Pairplot 그리기
sns.set(style="ticks")
sns.pairplot(df_selected, hue='cluster')
plt.show()                
                ''',language='python')
        image2_7 = Image.open("image/KMeans_2.png")
        st.image(image2_7, caption='2개 클러스터로 진행한 K-Means')
        image2_8 = Image.open("image/KMeans_5.png")
        st.image(image2_8, caption='5개 클러스터로 진행한 K-Means')
        st.markdown('''
        | Cluster | 0     | 1     | 2     | 3     | 4     |
        |---------|-------|-------|-------|-------|-------|
        | 2       | 9307  | 7036  |       |       |       |
        | 3       | 9237  | 4408  | 2698  |       |       |
        | 4       | 9192  | 3040  | 2367  | 1744  |       |
        | 5       | 9188  | 2170  | 2000  | 1536  | 1499  |
        ''')
        st.markdown('''
        군집 수를 바꿔가며 분석을 진행한 결과, 매출이 낮은 구간은 별 다른 변화 없이 매출이 높은 구간만 세분화되는 모습을 볼 수 있었습니다.
        이에 따라 군집 수는 2개로 나누어 매출이 높은 군집과 매출이 낮은 군집으로 구분하였습니다.
        ''')
        st.markdown('''
        | Silhouette Score | Davies-Bouldin Score |
        |----------|----------|
        | 0.4269096402569956 | 1.1408515114535012 |
        ''')
        st.markdown('''
        군집의 분할 정도를 나타내는 지표도 양호하게 나타났습니다.
        ''')

    elif RESEARCH3_choice == "4. 결과 확인":
        st.header("4. 결과 확인")
        st.markdown('-----')

        # 5.1 군집 분석 결과
        st.subheader("4.1 군집 분석 결과 해석 및 결론")
        image2_11 = Image.open("image/dot_density_synergy.png")
        st.image(image2_11, caption='군집 시각화')
        image2_9 = Image.open("image/KMeans_Cafe.png")
        st.image(image2_9, caption='군집별 카페 매출 분포')
        st.markdown('''
        | cluster | nums |
        |---------|------|
        | 0       | 1839 |
        | 1       | 1804 |
        ''')
        st.markdown('''
        **군집별 카페 분포의 해석**

        - 군집 0과 1은 각각 매출이 낮은 군집과 매출이 높은 군집을 나타냅니다.
        - 카페의 군집 위치 분석을 통해, 카페가 매출이 높은 군집과 낮은 군집에 골고루 분포하고 있음을 확인할 수 있었습니다.
        - 이는 카페가 위치한 지역의 특성, 경쟁 상황, 타겟 고객층 등 다양한 요인에 따라 카페의 성공 가능성이 달라질 수 있음을 의미합니다.
        ''')
        
        st.markdown('''
        **군집 분석 결론**
        - 군집 분석 결과는 카페와 다른 업종과의 상생 가능성을 평가하는 데 중요한 기초 자료로 활용됩니다.
        - 예를 들어, 매출이 높은 군집에 주로 위치한 다른 업종은 카페와의 상생 가능성이 높을 수 있습니다.
        ''')

    elif RESEARCH3_choice == "5. 시사점 및 제언":
        st.header("5. 시사점 및 제언")
        st.markdown('-----')

        # 6.1 상권별 카페와의 상생 정도 파악
        st.subheader("5.1 상권별 카페와의 상생 정도 파악")
        st.markdown('''
        **상권별 카페 상생 정도의 중요성**
        - 서울시 상권분석 서비스에 본 알고리즘을 통합함으로써, 특정 상권에서 카페와 상생 가능성이 높은 업종을 식별할 수 있으며 상생 가능한 정도를 백분율로 제공할 수 있습니다.
        ''')
        image2_10 = Image.open("image/적용방안사진.png")
        st.image(image2_10, caption='적용 방안')
        image2_12 = Image.open("image/Synergy Level Map.png")
        st.image(image2_12, caption='Synergy Level Map')
        
