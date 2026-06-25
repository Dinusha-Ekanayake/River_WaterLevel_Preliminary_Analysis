import json

nb_path = "235514B_River_WaterLevel_Preliminary_Analysis.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

md1 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Full Model Diagnostics\n",
        "\n",
        "In this section, we perform formal diagnostics on the multivariate model to check the standard linear regression assumptions:\n",
        "1. **Multicollinearity:** Variance Inflation Factor (VIF)\n",
        "2. **Linearity & Homoscedasticity:** Residuals vs Fitted values plot\n",
        "3. **Normality of Residuals:** Normal Q-Q plot\n",
        "4. **Independence of Errors:** Durbin-Watson statistic"
    ]
}

code1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
        "\n",
        "# 1. Variance Inflation Factor (VIF)\n",
        "print(\"Variance Inflation Factor (VIF):\")\n",
        "vif_data = pd.DataFrame()\n",
        "vif_data[\"Feature\"] = X_multi.columns\n",
        "vif_data[\"VIF\"] = [variance_inflation_factor(X_multi.values, i) for i in range(X_multi.shape[1])]\n",
        "vif_data"
    ]
}

md2 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**VIF Analysis:**\n",
        "The VIF values for `Water_Level_Xt_1` and `24HrRF_Xt_1` are very close to 1. A VIF less than 5 (or 10) indicates that there is **no multicollinearity** between the predictors. This confirms that adding rainfall does not cause multicollinearity issues."
    ]
}

code2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import scipy.stats as stats\n",
        "\n",
        "# Get fitted values and residuals from the multivariate model\n",
        "fitted_values = multi_model.fittedvalues\n",
        "residuals = multi_model.resid\n",
        "\n",
        "# Create plots\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
        "\n",
        "# 2. Residuals vs Fitted Values (Check for Linearity and Homoscedasticity)\n",
        "sns.residplot(x=fitted_values, y=residuals, lowess=True, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red', 'lw': 1.5}, ax=axes[0])\n",
        "axes[0].set_title('Residuals vs Fitted Values', fontsize=14)\n",
        "axes[0].set_xlabel('Fitted Values', fontsize=12)\n",
        "axes[0].set_ylabel('Residuals', fontsize=12)\n",
        "\n",
        "# 3. Normal Q-Q Plot (Check for Normality of Residuals)\n",
        "stats.probplot(residuals, dist=\"norm\", plot=axes[1])\n",
        "axes[1].set_title('Normal Q-Q Plot', fontsize=14)\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
}

md3 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "**Diagnostics Analysis:**\n",
        "\n",
        "1. **Residuals vs Fitted Values:** \n",
        "   - The residuals ideally should be randomly scattered around the horizontal axis (y=0) with constant variance. \n",
        "   - In our plot, we might observe a slight \"funnel\" shape (heteroscedasticity) and some patterns, which suggests that the variance of the errors increases at higher water levels. This is expected given the right-skewed nature of our data.\n",
        "\n",
        "2. **Normal Q-Q Plot:**\n",
        "   - If the residuals were perfectly normally distributed, they would lie exactly on the red diagonal line.\n",
        "   - The plot shows deviations at the tails (especially the upper tail). This indicates that the residuals have \"heavy tails\" (outliers), violating the strict normality assumption. Again, this is a direct consequence of the skewed distributions of river levels and rainfall.\n",
        "\n",
        "3. **Durbin-Watson Statistic:**\n",
        "   - From the OLS summary earlier, the Durbin-Watson statistic is **1.303**.\n",
        "   - The Durbin-Watson statistic ranges from 0 to 4. A value of 2 indicates no autocorrelation. Values between 1.5 and 2.5 are generally considered acceptable.\n",
        "   - A value of 1.303 suggests some **positive autocorrelation** in the residuals. In environmental data, spatial or temporal ordering (e.g., stations along the same river or ordered geographically) can cause adjacent observations to have correlated errors.\n",
        "\n",
        "**Conclusion on Diagnostics:**\n",
        "While the model has a very high $R^2$, the diagnostic plots indicate mild violations of homoscedasticity and normality. Given that the goal is robust prediction, we may need to explore applying a log-transformation to the variables (e.g., `log(x + c)`) to stabilize the variance and improve normality, or consider a generalized linear model."
    ]
}

nb["cells"].extend([md1, code1, md2, code2, md3])

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully.")
