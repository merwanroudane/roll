import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS, RollingWLS
import io
import base64
from matplotlib.figure import Figure
import seaborn as sns

st.set_page_config(page_title="Rolling Regression Analysis", layout="wide")


def download_link(object_to_download, download_filename, download_link_text):
    """Generate a link to download the provided object."""
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=True)

    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


def main():
    st.title("Rolling Regression Analysis")

    st.write("""
    This app performs rolling regression analysis using statsmodels. 
    Upload your data, select variables, and customize parameters to analyze how regression coefficients change over time.
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)

            # Display the dataframe
            st.subheader("Data Preview")
            st.dataframe(df.head())

            # Column selection
            st.subheader("Variable Selection")

            # Check if the dataframe has a datetime index
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            if not date_cols:
                date_cols = df.columns.tolist()

            date_column = st.selectbox("Select date column (if applicable)",
                                       options=['None'] + date_cols)

            if date_column != 'None':
                df = df.set_index(date_column)
                st.info(f"Set {date_column} as index")

            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

            # Select dependent variable (y)
            dependent_var = st.selectbox("Select dependent variable (y)",
                                         options=numeric_cols)

            # Select independent variables (X)
            remaining_cols = [col for col in numeric_cols if col != dependent_var]
            independent_vars = st.multiselect("Select independent variables (X)",
                                              options=remaining_cols,
                                              default=remaining_cols[:min(3, len(remaining_cols))])

            if not independent_vars:
                st.warning("Please select at least one independent variable.")
                return

            # Rolling regression parameters
            st.subheader("Rolling Regression Parameters")

            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider("Window size", min_value=5, max_value=min(500, len(df)),
                                        value=min(60, len(df) // 4), step=1)
                min_nobs = st.slider("Minimum observations", min_value=3,
                                     max_value=window_size, value=window_size, step=1)

            with col2:
                expanding = st.checkbox("Use expanding window", value=False)
                use_weights = st.checkbox("Use weighted least squares", value=False)

            # Advanced options
            with st.expander("Advanced Options"):
                cov_type = st.selectbox("Covariance type", options=["nonrobust", "HC0"],
                                        index=0)
                params_only = st.checkbox("Estimate parameters only (faster)", value=False)
                add_constant = st.checkbox("Add constant (intercept)", value=True)

                if use_weights:
                    weight_var = st.selectbox("Select weight variable",
                                              options=['Equal weights'] + numeric_cols)

            if st.button("Run Rolling Regression"):
                st.subheader("Rolling Regression Results")

                # Prepare data
                y = df[dependent_var]

                if add_constant:
                    X = sm.add_constant(df[independent_vars])
                    X_vars = ['const'] + independent_vars
                else:
                    X = df[independent_vars]
                    X_vars = independent_vars

                # Create weights if requested
                weights = None
                if use_weights and weight_var != 'Equal weights':
                    weights = df[weight_var]

                # Run regression
                with st.spinner("Running rolling regression..."):
                    try:
                        if use_weights:
                            model = RollingWLS(y, X, weights=weights, window=window_size,
                                               min_nobs=min_nobs, expanding=expanding)
                        else:
                            model = RollingOLS(y, X, window=window_size,
                                               min_nobs=min_nobs, expanding=expanding)

                        results = model.fit(cov_type=cov_type, params_only=params_only)

                        # Display coefficient table
                        st.subheader("Coefficient Statistics")

                        # Create a more complete summary of results
                        if not params_only:
                            mean_params = results.params.mean()
                            std_params = results.params.std()
                            min_params = results.params.min()
                            max_params = results.params.max()

                            summary_df = pd.DataFrame({
                                'Mean': mean_params,
                                'Std Dev': std_params,
                                'Min': min_params,
                                'Max': max_params
                            })

                            st.dataframe(summary_df)

                        # Plot results
                        st.subheader("Coefficient Plots")

                        fig_height = 5 * len(X_vars)
                        if not params_only:
                            fig = results.plot_recursive_coefficient(figsize=(10, fig_height))
                            st.pyplot(fig)
                        else:
                            # Manual plotting for params_only=True
                            fig = Figure(figsize=(10, fig_height))
                            for i, var in enumerate(X_vars):
                                ax = fig.add_subplot(len(X_vars), 1, i + 1)
                                results.params[var].plot(ax=ax)
                                ax.set_title(f"{var} coefficient")
                                ax.grid(True)
                            fig.tight_layout()
                            st.pyplot(fig)

                        # Display data for download
                        st.subheader("Download Results")

                        params_csv = results.params.to_csv().encode()
                        st.download_button(
                            label="Download coefficients as CSV",
                            data=params_csv,
                            file_name="rolling_regression_coefficients.csv",
                            mime="text/csv"
                        )

                        if not params_only:
                            # Create R-squared plot
                            st.subheader("R-squared Over Time")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            results.rsquared.plot(ax=ax)
                            ax.set_title("Rolling R-squared")
                            ax.grid(True)
                            st.pyplot(fig)

                            # Create residual plot for the last window
                            st.subheader("Residual Diagnostics (Last Window)")

                            # Get residuals from the last valid window
                            last_index = results.params.last_valid_index()
                            if last_index is not None:
                                window_end = df.index.get_loc(last_index)
                                window_start = max(0, window_end - window_size + 1)

                                y_window = y.iloc[window_start:window_end + 1]
                                X_window = X.iloc[window_start:window_end + 1]

                                if use_weights and weights is not None:
                                    w_window = weights.iloc[window_start:window_end + 1]
                                    window_model = sm.WLS(y_window, X_window, weights=w_window)
                                else:
                                    window_model = sm.OLS(y_window, X_window)

                                window_results = window_model.fit()

                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.scatter(window_results.fittedvalues, window_results.resid)
                                ax.axhline(y=0, color='r', linestyle='-')
                                ax.set_xlabel('Fitted values')
                                ax.set_ylabel('Residuals')
                                ax.set_title('Residuals vs Fitted')
                                ax.grid(True)
                                st.pyplot(fig)

                                # Create summary table for the last window
                                st.subheader("Last Window Summary")
                                summary_text = window_results.summary().as_text()
                                st.text(summary_text)

                            # Provide full results for download
                            r2_csv = results.rsquared.to_csv().encode()
                            st.download_button(
                                label="Download R-squared as CSV",
                                data=r2_csv,
                                file_name="rolling_regression_r2.csv",
                                mime="text/csv"
                            )

                    except Exception as e:
                        st.error(f"Error in regression: {str(e)}")
                        raise e

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")


if __name__ == "__main__":
    main()
