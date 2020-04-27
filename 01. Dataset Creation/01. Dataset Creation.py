import pandas as pd
import os

sRootPath = "./00. Raw dataset/BBC/bbc-fulltext/bbc"
sOutput = "./01. Dataset Creation/News_dataset_py.csv"

aColumns = ["File_name", "Content", "Category", "Complete_Filename"]
aRows = []
for sCategory in os.listdir(sRootPath):
    sPath = sRootPath + "/" + sCategory
    for sFile in os.listdir(sPath):
        with open(sPath + "/" + sFile, "r") as f:
            sContent = f.read()
        aRows.append([sFile, sContent, sCategory, sFile + "-" + sCategory])
d_Data = pd.DataFrame(aRows, columns=aColumns)
d_Data.to_csv(sOutput, index=False, encoding="utf-8")