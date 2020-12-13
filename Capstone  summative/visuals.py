import matplotlib.pyplot as plt


#function that diplays barcjart and line graph for  a dataframe

def barchart(p_df):
    min = int(p_df.efficiency.min() * 10) / 10

    def add_value_labels(ax, spacing=5):

        # For each bar: Place a label
        for rect in ax.patches:
            # Get X and Y placement of label from rect.
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            # Number of points between bar and label. Change to your liking.
            space = spacing
            # Vertical alignment for positive values
            va = 'bottom'

            # If value of bar is negative: Place label below bar
            if y_value < 0:
                # Invert space to place label below
                space *= -1
                # Vertically align label at top
                va = 'top'

            # Use Y value as label and format number with one decimal place
            label = "{:.3f}".format(y_value)

            # Create annotation
            ax.annotate(
                label,  # Use `label` as label
                (x_value, y_value),  # Place label at end of the bar
                xytext=(0, space),  # Vertically shift label by `space`
                textcoords="offset points",  # Interpret `xytext` as offset in points
                ha='center',  # Horizontally center label
                va=va)  # Vertically align label differently for
            # positive and negative values.


    fig = plt.figure(figsize=(9,5))


    ax1=p_df.plot(x='model',kind='bar', figsize=(15,5),title='Efficiency Analysis',width=0.7)
    ax1.set_ylabel('Efficiency')
    ax1.set_xlabel('Propeller Model')
    ax1.set_ylim(min)
    ax1.set_xticklabels(p_df.model,rotation=20);
    add_value_labels(ax1)

    p_df=p_df.round(decimals=3)
    ax2 = fig.add_subplot(122)
    font_size=12
    bbox=[0, 0, 1, 1]

    ax2.axis('off')

    mpl_table = ax2.table(cellText = p_df.values, rowLabels = p_df.index, bbox=bbox, colLabels=p_df.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    return {}



def lineplot(dff, wind_df):
    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_subplot(121)

    dff.plot(figsize=(15, 8), marker='x', title='Efficiency Foreccast')
    plt.ylabel('Efficency')
    # ax1.scatter(x=df['x'],y=df['y'])
    wind_df = wind_df.round(decimals=3)

    ax2 = fig.add_subplot(122)
    font_size = 10
    bbox = [0, 0, 1, 1]
    ax1.axis('off')
    ax2.axis('off')
    mpl_table = ax1.table(cellText=wind_df.values, rowLabels=wind_df.index, bbox=bbox, colLabels=wind_df.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    return {}
