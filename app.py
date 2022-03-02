import os
import pickle
import shap
import json
import requests
import alibi
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


@st.cache
def deserialization():
    my_directory = os.path.dirname(__file__)
    pickle_model_objects_path = os.path.join(my_directory, "interpretation_objects.pkl")
    with open(pickle_model_objects_path, "rb") as handle:
        explainer, features, feature_names = pickle.load(handle)
    return explainer, features, feature_names

# Load shap explainer
explainer, features, feature_names = deserialization()


@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df


# Load data
#path = "https://raw.githubusercontent.com/antoine-mle/OpenClassrooms-P7/main/dataframe.csv"
path = "https://raw.githubusercontent.com/antoine-mle/OpenClassrooms-P7/main/dataframe.csv?token=GHSAT0AAAAAABPW43BCGZO37532PFZSI4AIYQ7RWGQ"
df = load_data(path=path)


@st.cache
def split_data(df, num_rows):
    X = df.iloc[:, 2:]
    y = df["TARGET"]
    ids = df["SK_ID_CURR"]
    _, X_test, _, y_test, _, ids = train_test_split(
        X,
        y,
        ids,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_test = X_test.iloc[:num_rows, ]
    y_test = y_test.iloc[:num_rows, ]
    ids = list(ids[:num_rows, ])
    return X_test, y_test, ids


# Split data
X_test, y_test, ids = split_data(df=df, num_rows=1000)


@st.cache(allow_output_mutation=True)
def model_prediction(input):
    #url = "http://127.0.0.1:5000/predict"
    url = 'http://antoinemle.eu.pythonanywhere.com/predict'
    r = requests.post(url, json=input, timeout=120)
    return r.json()


def main():
    st.sidebar.header("Parameters:")
    page = st.sidebar.selectbox(
        "Choose an option:",
        ["Data Analysis", "Prediction", "Client Comparison", "Feature Importance"],
        index=0,
    )

    df_analysis = df.copy()
    for col in df_analysis.filter(like="DAYS").columns:
        df_analysis[col] = df_analysis[col].apply(lambda x: abs(x / 365))
    df_analysis.columns = df_analysis.columns.str.replace("DAYS", "YEARS")
    df_analysis["TARGET"] = df_analysis["TARGET"].astype(str)
    choice_list = list(df_analysis.iloc[:, 2:].columns)

    if page == "Data Analysis":
        st.title("Data Exploration")
        data_analysis = st.sidebar.selectbox(
            "Choose a type of analysis:",
            ["Univariate", "Bivariate", "Multivariate"],
            index=0,
        )

        if data_analysis == "Univariate":
            st.header("Univariate Analysis")
            options = st.multiselect(
                "Choose a variable to analyse",
                choice_list,
                ["AMT_INCOME_TOTAL", "AMT_CREDIT", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "YEARS_BIRTH"])

            if df_analysis[options].select_dtypes(include=["int64", "float64"]).shape[1] > 0:
                graphic_style = st.sidebar.radio(
                    "Select a graphic style for numerical features",
                    ("Histogram", "Box Plot"),
                    index=0,
                )

            if len(options) > 1:
                col1, col2 = st.columns(2)

            for i in range(len(options)):
                if df_analysis[options[i]].dtype == "object":
                    data = df_analysis.groupby("TARGET")[options[i]].value_counts().reset_index(name="percent")
                    data["percent"] = (data["percent"] / len(df_analysis) * 100).round(1)
                    fig = px.bar(
                        data,
                        x=options[i],
                        y="percent",
                        color="TARGET",
                        text_auto=True,
                        color_discrete_sequence=px.colors.qualitative.Pastel2,
                    )
                    if len(options) > 1:
                        if i % 2 == 0:
                            col1.plotly_chart(fig, use_container_width=True)
                        else:
                            col2.plotly_chart(fig, use_container_width=True)
                    else:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    if graphic_style == "Box Plot":
                        fig = px.box(
                            df_analysis,
                            y=options[i],
                            color="TARGET",
                            category_orders={"TARGET": ["0", "1"]},
                            color_discrete_sequence=px.colors.qualitative.Pastel2,
                        )
                    else:
                        fig = px.histogram(
                            df_analysis,
                            x=options[i],
                            color="TARGET",
                            category_orders={"TARGET": ["0", "1"]},
                            histnorm="percent",
                            nbins=10,
                            color_discrete_sequence=px.colors.qualitative.Pastel2,
                        )
                        fig.update_layout(bargap=0.1)
                    if len(options) > 1:
                        if i % 2 == 0:
                            col1.plotly_chart(fig, use_container_width=True)
                        else:
                            col2.plotly_chart(fig, use_container_width=True)
                    else:
                        st.plotly_chart(fig, use_container_width=True)

        elif data_analysis == "Bivariate":
            st.header("Bivariate Analysis")
            st.write("Choose two variables to analyse")
            col1, col2 = st.columns(2)

            feat_1 = col1.selectbox(
                "Variable 1",
                choice_list,
                index=choice_list.index("AMT_CREDIT"),
            )

            new_choice_list = [item for item in choice_list if item not in [feat_1]]
            try:
                index2 = new_choice_list.index("AMT_INCOME_TOTAL")
            except:
                index2 = 0
            feat_2 = col2.selectbox(
                "Variable 2",
                new_choice_list,
                index=index2
            )

            if (df_analysis[feat_1].dtype == "object" and df_analysis[feat_2].dtype == "object"):
                cont = df_analysis[[feat_1, feat_2]].pivot_table(
                    index=feat_1,
                    columns=feat_2,
                    aggfunc=len
                ).fillna(0)
                fig = px.imshow(
                    cont,
                    text_auto=True,
                    color_continuous_scale="teal"
                )
                st.plotly_chart(fig, use_container_width=True)

            elif (df_analysis[feat_1].dtype == "object" or df_analysis[feat_2].dtype == "object"):
                fig = px.box(
                    df_analysis,
                    x=feat_1,
                    y=feat_2,
                    color="TARGET",
                    category_orders={"TARGET": ["0", "1"]},
                )
                st.plotly_chart(fig, use_container_width=True)

            else:
                fig = px.scatter(
                    df_analysis,
                    x=feat_1,
                    y=feat_2,
                    color="TARGET",
                    hover_name="SK_ID_CURR",
                    category_orders={"TARGET": ["0", "1"]},
                    opacity=0.25,
                )

                fig.update_traces(
                    marker=dict(line=dict(width=0), size=3),
                    selector=dict(mode='markers')
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.header("Multivariate Analysis")
            graphic_style = st.sidebar.radio(
                "Select a graphic style for numerical features",
                ("Correlation Matrix", "Scatter Plot")
            )
            num_choice_list = list(df_analysis.iloc[:, 2:].select_dtypes(include=["int64", "float64"]).columns)

            if graphic_style == "Correlation Matrix":
                container = st.container()
                all_list = st.checkbox("Select all")
                if all_list:
                    options = container.multiselect(
                        "Choose several numerical variables to analyse:",
                        num_choice_list,
                        num_choice_list,
                    )
                else:
                    options = container.multiselect(
                        "Choose several numerical variables to analyse:",
                        num_choice_list,
                        ["AMT_INCOME_TOTAL", "AMT_CREDIT", "EXT_SOURCE_2", "EXT_SOURCE_3"],
                    )
                if len(options) > 0:
                    import plotly.io as pio
                    pio.templates.default = "none"
                    corr = df_analysis[["TARGET"] + options]
                    corr["TARGET"] = corr["TARGET"].astype(int)
                    corr = corr.corr()
                    mask = np.eye(corr.shape[0], dtype=bool)
                    corr = corr.mask(mask)
                    fig = px.imshow(
                        corr,
                        text_auto=True,
                        color_continuous_scale="RdBu_r"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Please select at least 1 feature")

            else:
                st.write("Choose three numerical variables to analyse")
                col1, col2, col3 = st.columns(3)
                feat_1 = col1.selectbox(
                    "Variable 1",
                    num_choice_list,
                    index=num_choice_list.index("AMT_ANNUITY"),
                )

                second_choice_list = [item for item in num_choice_list if item not in [feat_1]]
                try:
                    index2 = second_choice_list.index("AMT_CREDIT")
                except:
                    index2 = 0
                feat_2 = col2.selectbox(
                    "Variable 2",
                    second_choice_list,
                    index=index2,
                )

                third_choice_list = [item for item in num_choice_list if item not in [feat_1, feat_2]]
                try:
                    index3 = third_choice_list.index("AMT_INCOME_TOTAL")
                except:
                    index3 = 0
                feat_3 = col3.selectbox(
                    "Variable 3",
                    third_choice_list,
                    index=index3,
                )

                fig = px.scatter_3d(
                    df_analysis,
                    x=feat_1,
                    y=feat_2,
                    z=feat_3,
                    size_max=1,
                    color="TARGET",
                    hover_name="SK_ID_CURR",
                    category_orders={"TARGET": ["0", "1"]},
                    opacity=0.25)
                fig.update_traces(
                    marker=dict(line=dict(width=0), size=1),
                    selector=dict(mode='markers')
                )
                st.plotly_chart(fig, use_container_width=True)


    elif page == "Prediction":
        sorted_ids = sorted(ids)
        client_id = st.sidebar.selectbox(
            "Select a client ID:",
            sorted_ids,
        )
        id_idx = ids.index(client_id)
        client_input = X_test.iloc[[id_idx], :]

        st.title("Default Prediction")
        st.header("Make a prediction for client #{}".format(client_id))

        predict_button = st.button("Predict")
        if predict_button:
            client_input_json = json.loads(client_input.to_json())
            pred = model_prediction(client_input_json)["prediction"]
            proba = model_prediction(client_input_json)["probability"]
            # true_value = y_test.iloc[id_idx]
            if pred == 0:
                st.success("Loan granted 🙂 (refund probability = {}%)".format(proba))
                st.balloons()
            else:
                st.error("Loan not granted 😞 (default probability = {}%)".format(proba))

            with st.expander("Show feature impact:"):
                force_plot = shap.force_plot(
                    base_value=explainer.expected_value[0],
                    shap_values=explainer.shap_values[0][id_idx],
                    features=features[id_idx],
                    feature_names=feature_names,
                    matplotlib=True,
                    show=False,
                )
                st.pyplot(force_plot)

                decision_plot, ax = plt.subplots()
                ax = shap.decision_plot(
                    base_value=explainer.expected_value[0],
                    shap_values=explainer.shap_values[0][id_idx],
                    features=features[id_idx],
                    feature_names=feature_names,
                )
                st.pyplot(decision_plot)

        with st.expander("Show client information:"):
            df_client_input = pd.DataFrame(
                client_input.to_numpy(),
                index=["Information"],
                columns=client_input.columns,
            ).astype(str).transpose()
            st.dataframe(df_client_input)

    elif page == "Client Comparison":
        sorted_ids = sorted(ids)
        client_id = st.sidebar.selectbox(
            "Select a client ID:",
            sorted_ids,
        )

        st.title("Behavior comparison")

        col_choice_list = list(df_analysis.iloc[:, 2:].select_dtypes(include=["object"]).columns)
        num_choice_list = list(df_analysis.iloc[:, 2:].select_dtypes(include=["int64", "float64"]).columns)

        st.write("Choose categorical groups and numerical variables to compare client #{} with similar clients".format(client_id))
        col1, col2 = st.columns(2)

        feat_1 = col1.multiselect(
            "Categorical variables to identify groups",
            col_choice_list,
            ["CODE_GENDER", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"]
        )

        feat_2 = col2.multiselect(
            "Numerical variables",
            num_choice_list,
            ["AMT_CREDIT", "AMT_ANNUITY", "AMT_INCOME_TOTAL", "YEARS_BIRTH", "YEARS_EMPLOYED"]
        )

        df_group = df_analysis.copy()
        mask = (df_group[feat_1].values == df_group[df_group["SK_ID_CURR"] == client_id][feat_1].values)
        indices = np.where(((np.invert(mask) * 1).sum(axis=1)) == 0)[0]
        df_group = df_group.iloc[indices, :]
        df_comparison = pd.DataFrame(columns=feat_2)
        df_comparison.loc["Selected client"] = df_group[df_group["SK_ID_CURR"] == client_id][feat_2].mean()
        df_comparison.loc["Clients having refunded (Mean)"] = df_group[(df_group["TARGET"] == "0") & (df_group["SK_ID_CURR"] != client_id)][feat_2].mean()
        df_comparison.loc["Defaulted clients (Mean)"] = df_group[(df_group["TARGET"] == "1") & (df_group["SK_ID_CURR"] != client_id)][feat_2].mean()
        df_comparison = df_comparison.round(2).rename_axis("Group").reset_index()

        fig = px.parallel_coordinates(
            data_frame=df_comparison,
            dimensions=feat_2,
            color=(1 + df_comparison.index),
            range_color=[0.5, 3.5],
            color_continuous_scale=[(0.00, px.colors.diverging.Tealrose[0]),
                                    (0.33, px.colors.diverging.Tealrose[0]),
                                    (0.33, px.colors.diverging.Tealrose[3]),
                                    (0.66, px.colors.diverging.Tealrose[3]),
                                    (0.66, px.colors.diverging.Tealrose[-1]),
                                    (1.00, px.colors.diverging.Tealrose[-1])]
        )

        fig.update_layout(coloraxis_colorbar=dict(
            title="Group",
            tickvals=[1, 2, 3],
            ticktext=df_comparison["Group"],
            lenmode="pixels", len=100,
        ))

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show client group information:"):
            df_selected_client = pd.DataFrame(
                df_group[df_group["SK_ID_CURR"] == client_id][feat_1].to_numpy(),
                index=["Information"],
                columns=feat_1,
            ).astype(str)
            st.dataframe(df_selected_client)

    elif page == "Feature Importance":
        st.title("Feature importance for prediction")

        summary_plot, ax = plt.subplots()
        ax = shap.summary_plot(
            shap_values=explainer.shap_values,
            features=features,
            feature_names=feature_names,
            plot_type="bar",
        )
        st.pyplot(summary_plot)


if __name__ == "__main__":
    main()
