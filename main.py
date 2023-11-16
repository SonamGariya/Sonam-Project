from dataprocess import data_loader, data_preprocess
from analysis import variable_selection
from matplotlib import pyplot as plt
import seaborn as sns
import logging


matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.ERROR)

def main():
    
    data = data_loader("data/loan_data.csv")

    # Pie chart of Important Variables

    data_copy = data.copy()
    data_copy["pub.rec"] = data_copy["pub.rec"].apply(lambda x: "Have public records" if x > 0 else "No public records")
    pie(data_copy, x = "pub.rec" , x_label = "Public records")

    data_copy["delinq.2yrs"] = data_copy["delinq.2yrs"].apply(lambda x: "Have delinquency" if x > 0 else "No delinquency")
    pie(data_copy, x = "delinq.2yrs", x_label = "Delinquency")

    data_copy["inq.last.6mths"] = data_copy["inq.last.6mths"].apply(lambda x: "Have inquiry" if x > 0 else "No inquiry")
    pie(data_copy, x = "inq.last.6mths", x_label = "Inquiry")

    data_copy["not.fully.paid"] = data_copy["not.fully.paid"].apply(lambda x: "Not fully paid" if x == 1 else "Fully paid")
    pie(data_copy, x = "not.fully.paid", x_label = "Fully paid and Not fully paid")

    data_copy["credit.policy"] = data_copy["credit.policy"].apply(lambda x: "Non-Defaulter" if x == 1 else "Defaulter")
    pie(data_copy, x = "credit.policy", x_label = "Defaulters and Non-Defaulters")

    pie(data_copy, x = "purpose", x_label = "Purpose of loan")

    discrete_columns = ["pub.rec", "delinq.2yrs", "inq.last.6mths", "not.fully.paid", "credit.policy", "purpose"]
    non_discrete_columns = set(data.columns) - set(discrete_columns)

    print("Non-discrete columns are: \n", non_discrete_columns)
    print("Discrete columns are: \n", discrete_columns)

    for col in discrete_columns:
        percentage_cat_plot(data_copy, x = col, y = "credit.policy")

    for col in non_discrete_columns:
        boxplot(data, cols = col)

    for col in non_discrete_columns:
        plot_density(data, col = col, category = "credit.policy")

    plot_correlation_matrix(data[list(non_discrete_columns)+["credit.policy"]])

    plot_linear_regression(data, x = "fico", y = "int.rate")
    plot_linear_regression(data, x = "fico", y = "log.annual.inc")
    plot_linear_regression(data, x = "fico", y = "dti")
    plot_linear_regression(data, x = "dti", y = "log.annual.inc")

    X, y = data_preprocess(data, y_name = "credit.policy")

    all_var_df , sig_var_df, X, y = variable_selection(X, y, type = "logistic")

    print("All variables are: \n", all_var_df.to_string())



def boxplot(data, cols, col_x = "credit.policy"):
    """
    Plot boxplot in xkcd and save it to output folder
    :param data: data for plotting
    :param cols: cols to plot
    :param col_x: in respect to which column, boxplot will be plotted
    :return: None
    """

    assert col_x in data.columns, "{} is not present in data".format(col_x)

    with plt.xkcd():
        sns.set(style = "whitegrid")
        sns.boxplot(x = col_x, y = cols, data = data)
        filename = "output/boxplot_" + col_x + ".png"
        plt.savefig(filename)
        plt.show()

def pie(data , x = None , x_label = None):
    """
    Plot pie chart in xkcd and save it to output folder
    :param data: data for plotting
    :param x: column name for which pie chart will be plotted
    :return: None
    """
    if x is None:
        raise ValueError("x must be provided")
    assert x in data.columns, "{} is not present in data".format(x)

    with plt.xkcd():
        plt.pie(data[x].value_counts(), labels = data[x].value_counts().index, autopct = "%1.1f%%")
        title_name = f"Share of {x_label} in P2P market"
        plt.title(title_name)
        filename = "output/pie_" + x + ".png"
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()


def percentage_cat_plot(data , x , y):
    """
    Plot percentage catplot in xkcd and save it to output folder
    :param data: data for plotting
    :param x: column name for which catplot will be plotted
    :param y: column name for which catplot will be plotted
    :return: None
    """
    if x is None:
        raise ValueError("x must be provided")
    assert x in data.columns, "{} is not present in data".format(x)


    if y is None:
        raise ValueError("y must be provided")
    assert y in data.columns, "{} is not present in data".format(y)

    df =  data.groupby(x)[[y]].value_counts(normalize = True).rename("percentage").mul(100).reset_index()

    with plt.xkcd():
        g = sns.catplot(x = x, y = "percentage", hue = y, data = df, kind = "bar", height = 5, aspect = 2)
        g.ax.set_ylim(0, 100)
        plt.tight_layout()
        filename = "output/catplot_" + x + ".png"
        plt.savefig(filename)
        plt.show()

def plot_correlation_matrix(data):
    """
    Plot correlation matrix in xkcd and save it to output folder
    :param data: data for plotting
    :return: None
    """
    with plt.xkcd():
        plt.figure(figsize = (10, 10))
        sns.heatmap(data.corr(), annot = True, cmap = "coolwarm")
        plt.tight_layout()
        plt.savefig("output/correlation_matrix.png")
        plt.show()

def plot_density(data, col , category = None):
    """
    Plot density plot in xkcd and save it to output folder
    :param data: data for plotting
    :param col: column name for which density plot will be plotted
    :param category: category name for which density plot will be plotted
    :return: None
    """
    if col is None:
        raise ValueError("col must be provided")
    assert col in data.columns, "{} is not present in data".format(col)

    if category is None:
        raise ValueError("category must be provided")
    assert category in data.columns, "{} is not present in data".format(category)

    with plt.xkcd():
        sns.kdeplot(data , x = col , hue = category)
        plt.tight_layout()
        filename = "output/density_" + col + ".png"
        plt.savefig(filename)
        plt.show()

def plot_linear_regression(data, x, y):
    """
    Plot linear regression in xkcd and save it to output folder
    :param data: data for plotting
    :param x: column name for which linear regression will be plotted
    :param y: column name for which linear regression will be plotted
    :return: None
    """
    if x is None:
        raise ValueError("x must be provided")
    assert x in data.columns, "{} is not present in data".format(x)

    if y is None:
        raise ValueError("y must be provided")
    assert y in data.columns, "{} is not present in data".format(y)

    with plt.xkcd():
        sns.lmplot(x = x, y = y, data = data)
        plt.tight_layout()
        filename = "output/linear_regression_" + x + y + ".png"
        plt.savefig(filename)
        plt.show()




if __name__ == '__main__':
    main()



















