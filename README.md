## Crimes Before/During The Pandemic

You can use the [editor on GitHub](https://github.com/Stevenh825-git/Crimes-Before-During-Pandemic/edit/main/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Stevenh825-git/Crimes-Before-During-Pandemic/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.

###Identifying whether a PCA analysis can be done

The goal is to see whether a 2 dimensional graph is enough to showcase patterns in the data using PCA. 

```markdown
#Represent the categorical data with numbers
df = pd.get_dummies(Hate_Crimes['Offense Category'])

#Center the data
cntr_offense = df - df.mean(axis=0)  

#Obtain 's', the one-dimensional array SVD produces
U, s, Vt = svd(cntr_offense, full_matrices=False) #Apply SVD

#Plotting the array squared will showcase the dimensions required to capture variance
plt.plot(s**2);
plt.suptitle('Dimensions Needed to Capture Variance', fontsize=20)
plt.savefig('DimensionsForPCA.png', bbox_inches='tight')
plt.show()

```
This was the code used to generate the following graph:

![DimensionPCA](/Crimes-Before-During-Pandemic/assets/css/DimensionsForPCA.png)

As a result, 2 dimensions is capable of showcasing the data's patterns. Or more specifically, 2 dimensions can be used to determine the spread of the different types of Hate Crimes. 


###Performing a PCA Analysis on the Data

![PCAForHateCrimes](/Crimes-Before-During-Pandemic/assets/css/PCAForHateCrimes.png)

```markdown
#Compute the first two principal components needed for PCA
pcs = U @ np.diag(s)
pc1 = pcs[:, 0]
pc2 = pcs[:, 1]

#Create a dataframe to add the principle components to the original data
case2d = pd.DataFrame({
    'case': df.index,
    'pc1': pcs[:, 0],
    'pc2': pcs[:, 1]
}).merge(Hate_Crimes, left_on='case', right_on=Hate_Crimes.index)

#Thsi function will be used to remove the data from completely overlapping
def jitter_df(df, x_col, y_col):
    x_jittered = df[x_col] + np.random.normal(scale=0.04, size=len(df))
    y_jittered = df[y_col] + np.random.normal(scale=0.06, size=len(df))
    return df.assign(**{x_col: x_jittered, y_col: y_jittered})


sns.scatterplot(data = jitter_df(case2d, 'pc1', 'pc2'),
                x="pc1", y="pc2", hue="Offense Category", alpha=(0.9));

plt.legend(bbox_to_anchor=(1.5, 1), loc='upper right')
plt.suptitle('PCA For Hate Crimes', fontsize=20)
plt.savefig('PCAForHateCrimes.png', bbox_inches='tight')
plt.show()
```

![Anti Asian](/Crimes-Before-During-Pandemic/assets/css/AntiAsian.png)

![Crimes During Pandemic](/Crimes-Before-During-Pandemic/assets/css/CrimesDuringPandemic.png)

![Crimes Before Pandemic](/Crimes-Before-During-Pandemic/assets/css/CrimesPrePandemic.png)

![Crimes Before and During Pandemic](/Crimes-Before-During-Pandemic/assets/css/CrimesPreDuringPandemic.png)


![GenderBeforeDuring](/Crimes-Before-During-Pandemic/assets/css/GenderBeforeDuring.png)

![HateCrimeLogisticModel1](/Crimes-Before-During-Pandemic/assets/css/HateCrimeLogisticModel1.png)

![HateCrimeLogisticModel2](/Crimes-Before-During-Pandemic/assets/css/HateCrimeLogisticModel2.png)

![HateCrimesBeforeDuringPandemic](/Crimes-Before-During-Pandemic/assets/css/HateCrimesBeforeDuringPandemic.png)

![HateCrimesPerBorough](/Crimes-Before-During-Pandemic/assets/css/HateCrimesPerBorough.png)


![PrecinctCrimesBorough](/Crimes-Before-During-Pandemic/assets/css/PrecinctCrimesBorough.png)



![TypeofRaceHateCrimes](/Crimes-Before-During-Pandemic/assets/css/TypeofRaceHateCrimes.png)



![Pie1](/Crimes-Before-During-Pandemic/assets/css/Pie1.png)

![Pie2](/Crimes-Before-During-Pandemic/assets/css/Pie2.png)



![PrecinctCrimesOccurance](/Crimes-Before-During-Pandemic/assets/css/PrecinctCrimesOccurance.png)

![TypeofHateCrimes](/Crimes-Before-During-Pandemic/assets/css/TypeofHateCrimes.png)


![WhetherCrimeAntiAsian](/Crimes-Before-During-Pandemic/assets/css/WhetherCrimeAntiAsian.png)

![WhetherRaceCrimeAntiAsian](/Crimes-Before-During-Pandemic/assets/css/WhetherRaceCrimeAntiAsian.png)
