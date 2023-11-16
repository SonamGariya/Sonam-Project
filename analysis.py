
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.discrete.discrete_model import Logit

def variable_selection(X,y,type = "logistic"):
    """
    Fit a logistic or linear regression and select the best variables
    which have p-value less than 0.05
    :param X: a dataframe of independent variables
    :param y: dependent variable
    :param type: Linear or logistic regression
    :return: A dataframe with significant variables, X, y
    """
    # fit a logistic regression from statssmodels
    column_names = X.columns
    X = X.astype(float)

    if type == "logistic":
        model = Logit(y, X).fit()
    elif type == "linear":
        model = ols("y ~ .", data = pd.concat([X, y], axis = 1)).fit()



    with open("output/model_summary.txt", "w") as f:
        f.write(str(model.summary()))


    p_values = model.pvalues

    sig_var = X.columns[p_values < 0.05]
    sig_p_values = p_values[p_values < 0.05]
    sig_var_df = pd.DataFrame({"Variable": sig_var, "p-value": sig_p_values})

    all_var_df = pd.DataFrame({"Variable": column_names, "p-value": p_values , "Significant": p_values < 0.05 , "Coefficient": model.params})


    with open("output/sig_var.txt", "w") as f:
        f.write(str(sig_var_df))
    print("\n-------------------------------------------------------------------------\n")
    print("Significant variables are: \n", sig_var_df)
    print("\n-------------------------------------------------------------------------\n")
    X = X[sig_var]
    return all_var_df , sig_var_df , X , y

