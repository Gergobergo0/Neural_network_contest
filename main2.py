import pandas as pd

# Eredeti CSV fájl beolvasása
input_file = 'solution.csv'  # Cseréld le a fájl nevét az eredeti CSV fájl nevére
output_file = 'solution2.csv'

# CSV betöltése
df = pd.read_csv(input_file)

# Az "Predicted" oszlop abszolút értéke és egész számra kerekítése
df['Predicted'] = df['Predicted'].apply(lambda x: round(abs(float(x))))

# Az eredmények mentése egy új CSV fájlba az eredeti formátum szerint
df.to_csv(output_file, index=False)




