import pandas as pd


def analyze():
    df = pd.read_csv('ml_data.csv')
    max1 = df.loc[df['y0'].idxmax()]
    max2 = df.loc[df['y1'].idxmax()]
    min1 = df['y0'].min()
    min2 = df['y1'].min()

    print("max y0: "+str(max1))
    print("min y0: "+str(min1))
    print("max y1: "+str(max2))
    print("min y1: "+str(min2))

if __name__ == '__main__':
    analyze()
