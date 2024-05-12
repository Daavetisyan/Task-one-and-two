import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('vou_100.csv')
redshift = pd.read_csv('redshifts.csv')

#1
redshift.rename(columns={'name': 'source_name'}, inplace=True)
redshift_no_duplicates = redshift.drop_duplicates('source_name')
redshift_grouped = redshift_no_duplicates.groupby('source_name').sum().reset_index()

data_merged = pd.merge(data, redshift_grouped, on='source_name', how='left')
print(data_merged)

#2
print(data_merged.describe())
print(data_merged.columns)
print(data_merged['frequency'].isna().sum())
print(data_merged['nufnu_upper'].isnull().sum())
print((data_merged['start_time'] == data_merged['end_time']).sum())
print(data[data_merged['start_time'] != data_merged['end_time']])


for column in data_merged.columns:
    nan_values = data_merged[column].isna().sum()
    print(f"Nan values count in column '{column}'")
    print(nan_values)

for column in data_merged.columns:
    null_values = (data_merged[column] == 0.0).sum()
    print(f"Null values count in column '{column}'")
    print(null_values)


print(data_merged['nufnu'].value_counts())
print((data_merged['nufnu'] == 0.0).sum())
print((data_merged['nufnu_upper'] == 0.0).sum())
print((data_merged['nufnu_lower'] == 0.0).sum())
print(((data_merged['nufnu'] == 0) & (data_merged['nufnu_upper'] == 0) & (data_merged['nufnu_lower'] == 0)).sum())
print(data_merged['Catalog'].unique())

#nufnu_err = (nufnu_upper - nufnu_lower) / 2
mask = (data_merged['nufnu'] == 0) & (data_merged['nufnu_upper'] == 0) & (data_merged['nufnu_lower'] == 0)
data_merged = data_merged[~mask]
print(data_merged['flag'].unique())
mask_1 = (data_merged['nufnu'] == 0) & (data_merged['nufnu_lower'] == 0)
data_merged = data_merged[~mask_1]

#3
import vispy.scene as vs
import astropy.units as u
from astropy.visualization import quantity_support

plt.figure(figsize=(8, 6))
sns.histplot(data=data_merged, x='nufnu', bins=30, kde=True)
plt.title('Distribution of nufnu')
plt.xlabel('nufnu')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data_merged, x='frequency', y='nufnu')
plt.title('Scatter plot of frequency vs nufnu')
plt.xlabel('Frequency')
plt.ylabel('nufnu')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=data_merged, x='flag')
plt.title('Distribution of flag')
plt.xlabel('Flag')
plt.ylabel('Count')
plt.show()


with quantity_support():
    fig, ax = plt.subplots()
    ax.plot(data['frequency'] * u.Hz, data['nufnu'] * u.Unit('erg / (cm2 s)'))
    ax.set_xlabel('Frequency')
    ax.set_ylabel(r'$\nu F_\nu$')
    ax.set_title('Spectral Energy Distribution')
    plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points
ax.scatter(data_merged['nufnu'], data_merged['nufnu_upper'], data_merged['nufnu_lower'])


ax.set_xlabel('nufnu')
ax.set_ylabel('nufnu_upper')
ax.set_zlabel('nufnu_lower')
ax.set_title('3D Scatter Plot of nufnu, nufnu_upper, and nufnu_lower')

plt.show()


plt.figure(figsize=(10, 6))
for catalog, group in data_merged.groupby('Catalog'):
    plt.plot(group['start_time'], group['nufnu'], marker='o', label=catalog)
plt.xlabel('Start Time')
plt.ylabel('Flux (nufnu)')
plt.title('Flux of Blazars Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#unique_catalogs = data_merged['Catalog'].unique()
#for catalog in unique_catalogs:
    #plt.figure(figsize=(10, 6))
    #catalog_data = data_merged[data_merged['Catalog'] == catalog]
    #plt.plot(catalog_data['start_time'], catalog_data['nufnu'], marker='o')
    #plt.xlabel('Start Time')
    #plt.ylabel('Flux (nufnu)')
    #plt.title(f'Flux of Blazars Over Time - {catalog}')
    #plt.grid(True)
    #plt.xticks(rotation=45)
    #plt.tight_layout()
    #plt.show()


UL = data_merged[data_merged["flag"] == "UL"]
det = data_merged[data_merged["flag"] == "det"]

plt.figure(figsize=(10, 6))
for catalog, group in UL.groupby('Catalog'):
    plt.plot(group['start_time'], group['nufnu'], marker='o', label=catalog)
plt.xlabel('Start Time')
plt.ylabel('Flux (nufnu)')
plt.title('Flux of Blazars Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#plt.figure(figsize=(10, 6))
#for catalog, group in det.groupby('Catalog'):
    #plt.plot(group['start_time'], group['nufnu'], marker='o', label=catalog)
#plt.xlabel('Start Time')
#plt.ylabel('Flux (nufnu)')
#plt.title('Flux of Blazars Over Time')
#plt.legend()
#plt.grid(True)
#plt.xticks(rotation=45)
#plt.tight_layout()
#plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data_merged["redshift"], data_merged["nufnu"], label="nufnu", alpha=0.5)
plt.xlabel("Redshift")
plt.ylabel("nufnu")
plt.title("nufnu vs. Redshift")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data_merged["redshift"], data_merged["nufnu_upper"], label="nufnu_upper", alpha=0.5)
plt.xlabel("Redshift")
plt.ylabel("nufnu_upper")
plt.title("nufnu_upper vs. Redshift")
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(data_merged["redshift"], data_merged["nufnu_lower"], label="nufnu_lower", alpha=0.5)
plt.xlabel("Redshift")
plt.ylabel("nufnu_lower")
plt.title("nufnu_lower vs. Redshift")
plt.legend()
plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(data_merged['nufnu_err'], data_merged['nufnu_upper'], data_merged['nufnu_lower'])

ax.set_xlabel('nufnu')
ax.set_ylabel('nufnu_upper')
ax.set_zlabel('nufnu_lower')
ax.set_title('3D Scatter Plot of nufnu, nufnu_upper, and nufnu_lower')

plt.show()


plt.figure(figsize=(8, 6))
plt.scatter(data_merged["nufnu_err"], data_merged["nufnu_lower"], label="nufnu_lower", alpha=0.5)
plt.xlabel("nufnu_err")
plt.ylabel("nufnu_lower")
plt.title("nufnu_lower vs. nufnu_err")
plt.legend()
plt.show()

sns.set(style="whitegrid")

plt.figure(figsize=(8, 6))
g = sns.FacetGrid(data=data_merged, height=4, aspect=10)
g.map_dataframe(sns.kdeplot, "redshift", "nufnu", shade=True, alpha=1, lw=1.5, bw_adjust=0.2)
g.set_axis_labels("Redshift", "nufnu")
g.fig.suptitle("nufnu vs. Redshift (Ridge Plot)")
plt.show()

plt.figure(figsize=(8, 6))
g = sns.FacetGrid(data=data_merged, height=4, aspect=10)
g.map_dataframe(sns.kdeplot, "redshift", "nufnu_upper", shade=True, alpha=1, lw=1.5, bw_adjust=0.2)
g.set_axis_labels("Redshift", "nufnu_upper")
g.fig.suptitle("nufnu_upper vs. Redshift (Ridge Plot)")
plt.show()

plt.figure(figsize=(8, 6))
g = sns.FacetGrid(data=data_merged, height=4, aspect=10)
g.map_dataframe(sns.kdeplot, "redshift", "nufnu_lower", shade=True, alpha=1, lw=1.5, bw_adjust=0.2)
g.set_axis_labels("Redshift", "nufnu_lower")
g.fig.suptitle("nufnu_lower vs. Redshift (Ridge Plot)")
plt.show()
