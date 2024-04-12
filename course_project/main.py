def main():
    print("Building linear regression model")
    from line_reg_model import plt as line_reg_plt
    line_reg_plt.show()
    print("Building decision tree model")
    from decision_tree_model import plt as decision_tree_plt
    decision_tree_plt.show()
    print("Building random forest model")
    from random_forest_model import plt as random_forest_plt
    random_forest_plt.show()


if __name__ == "__main__":
    main()
