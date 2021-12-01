# Number of Crimes Before/During The Pandemic
### Name & Profile
Name: Steven Hernandez

[LinkedIn](https://www.linkedin.com/in/steven-hernandez-736817199/)

[Github](https://github.com/Stevenh825-git)

[Website Link](https://stevenh825-git.github.io/Crimes-Before-During-Pandemic/)

### Completed:
-Included all of the visualizations made
-Added code for most of the visualizations
-Grouped the visualization by the topic they are discussing
-Tried to explain what the visualizations show
-Added links to each dataset, linkedin, and github

### Planning to Complete:

-Add the abstract to the beginning of the website
-Include a section at the end talking about the final results
-Include code and discussion for the pie charts
-Further elaborate some of the visualizations
-Spell Check

### Datesets Used:

[NYPD Hate Crimes](https://data.cityofnewyork.us/Public-Safety/NYPD-Hate-Crimes/bqiq-cu78)

[NYPD Arrest Data Year-to-Date](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrest-Data-Year-to-Date-/uip8-fykc)

[NYPD Arrest Data Historic](https://data.cityofnewyork.us/Public-Safety/NYPD-Arrests-Data-Historic-/8h9b-rp9u)



### Identifying whether a PCA analysis can be done

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


### Performing a PCA Analysis on the Data

Now that we now this, we can generate the principal components from performaing matrix multiplication with U and S; where they were both created from the SVD funciton earlier.

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
This code will create:

![PCAForHateCrimes](/Crimes-Before-During-Pandemic/assets/css/PCAForHateCrimes.png)

As a result, the data is grouped into 4 clusters. This visualization showcases that there are 3 main groups of Hate Crimes: Sexual Orientation, Religion, and Race. However, it also shows that there are plenty of other Hate Crimes that do not fall within these categories. 

### Types of Hate Crimes

Now that we see the pattern of the data, it is important to view how many of each type of Hate Crimes from 01-01-2019 to 09-30-2021 there actually are. 

Using the following code:
```markdown

#Can group by Offense Category and count how many of type of hate crime there are

amount_crimes = Hate_Crimes.groupby('Offense Category').agg(
       Number_of_Cases=pd.NamedAgg(column="Full Complaint ID", aggfunc="count"))

hate_crime_graph = sns.barplot(x =amount_crimes.index,y="Number_of_Cases", data=amount_crimes)
hate_crime_graph.set_xticklabels(hate_crime_graph.get_xticklabels(), rotation=45, ha="right")

plt.ylabel("Number of Cases",fontsize=14)
plt.xlabel("Offense Category", fontsize=14)

hate_crime_graph.set_title("Different Types of Hate Crimes(2019-2021)", fontsize=20)
plt.savefig('TypeofHateCrimes.png', bbox_inches='tight')
plt.show()
```

![TypeofHateCrimes](/Crimes-Before-During-Pandemic/assets/css/TypeofHateCrimes.png)

This graph showcases that similarly to the PCA, religion, race, and sexual orientation were the 3 biggest motivators for hate crimes. However, it also shows that there were still plenty of ethnicity and gender motivated crimes. 

### Different Race Motivated Hate Crimes

Now that it is clear there are plenty of differnt Hate Crimes, it is important to understand these are all categories. As a result, there are plenty of different type of race motivated crimes. Therefore, it is importnat to see which race was the most targeted from 2019-2021. 

```markdown

#Get the rows exclusive to Race/Color and Ethnicity
race_ethni_crimes = Hate_Crimes.loc[(Hate_Crimes['Offense Category'] == 'Race/Color') | (Hate_Crimes['Offense Category'] == 'Ethnicity/National Origin/Ancestry')]

#Group the data by Bias Motive Descirption, which is the reason for the Hate Crime. Also, count how many cases of each group there are. 
amount_race = race_ethni_crimes.groupby('Bias Motive Description').agg(
       Number_of_Cases=pd.NamedAgg(column="Full Complaint ID", aggfunc="count"))

race_graph = sns.barplot(x =amount_race.index,y="Number_of_Cases", data=amount_race)
race_graph.set_xticklabels(race_graph.get_xticklabels(), rotation=45, ha="right")
plt.xlabel("Motive", fontsize=18)
plt.ylabel("Number of Cases", fontsize=18)

race_graph.set_title("Different Types of Race Motivated Hate Crimes(2019-2021)",
                     fontsize=20)
plt.savefig('TypeofRaceHateCrimes.png', bbox_inches='tight')
plt.show()
```

![TypeofRaceHateCrimes](/Crimes-Before-During-Pandemic/assets/css/TypeofRaceHateCrimes.png)

This showcases that from 2019 to 2021, there have been an overwhelming amount of Anti-Asian motivated Hate Crimes.In fact, other than Anti-Black crimes, Anti-Asian Hate Crimes dominate the graph. However, how much of these cases had occured before the pandemic?

### Anti-Asian Crimes Before and During the Pandemic

Before we can investigate, the data needs to be cleaned. 

```markdown
#Set the data found in Record Create Date to datetime objects. 
race_ethni_crimes['Record Create Date'] = pd.to_datetime(race_ethni_crimes['Record Create Date'])

#Obtain all of the Hate Crimes that were targetting Asians. 
anti_asian = race_ethni_crimes.loc[race_ethni_crimes['Bias Motive Description'] == "ANTI-ASIAN"]

#This function will determine whether the dates within a column happened before Covid-19 was declared by WHO, or during the Pandemic.
def prepostPandemic(date):
    
    temp = date['Record Create Date']
    temp2 = []
    start = "03/11/2020"
    start = datetime.strptime(start, "%m/%d/%Y")
    
    for day in temp:
        if day < start:
            temp2.append("Before")
        else:
            temp2.append("During")
    return temp2

#A new column will hold the information whether a crime occured before or during the pandemic
anti_asian['Before/During Pandemic'] = prepostPandemic(anti_asian)
```


```markdown
#Now it is easier to group the crimes by their occurance, and count how many of each there are
temp = anti_asian.groupby('Before/During Pandemic').agg(
       Number_of_Cases=pd.NamedAgg(column="Full Complaint ID", aggfunc="count"))


asian_cases = sns.barplot(x =temp.index,y="Number_of_Cases", data=temp)
xes = ["01/15/2019 - 03/10/2020", "03/11/2020 - 09/23/2021"]
asian_cases.set_xticklabels(xes,fontsize=18)

asian_cases.set_title("Amount of Anti-Asian Cases Before & During the Pandemic",
                      fontsize=20)

plt.xlabel("Before/During Pandemic", fontsize=18)
plt.ylabel("Number of Cases", fontsize=18)

plt.savefig('AntiAsian.png', bbox_inches='tight')
plt.show()
```
![Anti Asian](/Crimes-Before-During-Pandemic/assets/css/AntiAsian.png)

This graph showcases a huge difference between the amount of cases before and during the pandemic. While the amount of months for each category is unequal, there were only about 6 crimes targetting Asians within 14 months versus almost 160 crimes within 18 months. As a result, it is safe to say there has been an increase in stigmatization towards Asians during the Pandemic. 

### Amout of Hate Crimes Per Borough 
It is important to investigate which borough experienced the most crimes. For that reason, the following function is required:

```markdown
#Function to replace any complicated wording for boroughs
def cleanBorough(borough):

    if borough == 'PATROL BORO BKLYN NORTH':
        return 'Brooklyn'
    elif borough == 'PATROL BORO BKLYN SOUTH':
        return 'Brooklyn'
    elif borough == 'PATROL BORO BRONX':
        return 'Bronx'
    elif borough == 'PATROL BORO MAN NORTH':
        return 'Manhattan'
    elif borough == 'PATROL BORO MAN SOUTH':
        return 'Manhattan'
    elif borough == "PATROL BORO QUEENS NORTH":
        return "Queens"
    elif borough == "PATROL BORO QUEENS SOUTH":
        return "Queens"
    else:
        return 'Staten Island'
    
```
Originally in the dataset, Hate Crimes were seperated by the borough they occurred in. However, it also distinguished between North and South sections of Manhattan, Brooklyn, and Queens. To make things easier, I grouped these sections together. 

```markdown
#Clean the data
Hate_Crimes['Patrol Borough Name'] = Hate_Crimes['Patrol Borough Name'].apply(
                                    cleanBorough)

#Group by month and borough, while counting how many cases for each group
month_borough = Hate_Crimes.groupby(['Patrol Borough Name','Month Number']).agg(
       Number_of_Cases=pd.NamedAgg(column="Full Complaint ID", aggfunc="count"))
month_borough = month_borough.reset_index()

#Create the graph
month_boro_graph = sns.lmplot(x="Month Number", y="Number_of_Cases", hue="Patrol Borough Name", data=month_borough, fit_reg=(False))

month_boro_graph.set(ylabel = "Number of Cases")
plt.title("Amount of Hate Crimes per Borough by Month")


#Plot the lines connecting the dots
bronx = month_borough.loc[(month_borough['Patrol Borough Name'] == 'Bronx')]
plt.plot(bronx["Month Number"],bronx["Number_of_Cases"], 'b')

brooklyn = month_borough.loc[(month_borough['Patrol Borough Name'] == 'Brooklyn')]
plt.plot(brooklyn["Month Number"],brooklyn["Number_of_Cases"], 'y')

queens = month_borough.loc[(month_borough['Patrol Borough Name'] == 'Queens')]
plt.plot(queens["Month Number"],queens["Number_of_Cases"], 'r')

manhattan = month_borough.loc[(month_borough['Patrol Borough Name'] == 'Manhattan')]
plt.plot(manhattan["Month Number"],manhattan["Number_of_Cases"], 'g')

staten = month_borough.loc[(month_borough['Patrol Borough Name'] == 'Staten Island')]
plt.plot(staten["Month Number"],staten["Number_of_Cases"], 'm')

plt.savefig('HateCrimesPerBorough.png', bbox_inches='tight')
plt.show()
```
This would generate an lmplot where there are 5 colorcoded lines corresponding to one of the 5 boroughs. 

![HateCrimesPerBorough](/Crimes-Before-During-Pandemic/assets/css/HateCrimesPerBorough.png)

This lmplot showcases that the Hate Crime dataset, which spans from 01-01-2019 to 09-30-2021, contains a lot of data for Brookklyn and Manhattan, whereas the Bronx and Staten Island faced the least hate crimes. It is worth noting that the decrease in cases for all 5 boroughs during October to December most likely stems that there weren't any data yet for those months within 2021, thus a decrease in cases occuring during those months overall. 

This lead to the question: How many of the Hate Crimes within this graph occured before the Pandemic?

```markdown
#Goal is to group the data by the borough, but also if it occured before or during the pandemic 

#Need to set the dates to a datetime object
Hate_Crimes['Record Create Date'] = pd.to_datetime(Hate_Crimes['Record Create Date'])
#Apply the previous function used for Anti-Asian crimes to the entire dataset
Hate_Crimes['Before/During Pandemic'] = prepostPandemic(Hate_Crimes)

#Group by the Borough and whether the crime occurred before or during the pandemic
preduring_borough = Hate_Crimes.groupby(['Patrol Borough Name','Before/During Pandemic']).agg(
       Number_of_Cases=pd.NamedAgg(column="Full Complaint ID", aggfunc="count"))
preduring_borough = preduring_borough.reset_index()

#create the graph
plt.figure(figsize=(8,4))
boroughPrepost = sns.barplot(x ="Patrol Borough Name",y="Number_of_Cases",
                          hue = "Before/During Pandemic", data=preduring_borough)
title = "(01/15/2019 - 03/10/2020) vs (03/11/2020 - 09/30/2021)"

boroughPrepost.set(xlabel = "Borough",ylabel = "Number of Cases")
boroughPrepost.set_title(title)
plt.suptitle("Amount of Hate Crimes Before & During the Pandemic")
plt.savefig('HateCrimesBeforeDuringPandemic.png', bbox_inches='tight')
plt.show()
```
![HateCrimesBeforeDuringPandemic](/Crimes-Before-During-Pandemic/assets/css/HateCrimesBeforeDuringPandemic.png)

This graph shows clearly how many Hate Crimes each borough experienced. One can easily see that Brooklyn and Manhattan are completely dominating the other boroughs, regardless if the case occured before or during the pandemic. 

It is also worth noting that Manhattan experiences way more Hate Crimes during the Pandemic than any other borough. 

### Logistic Regression Model First Attempt 

The goal is to see if a logistic regression could be used to predict whether a Hate Crime's motive is Anti-Asian using the month in the year.

```markdown
#First needed to use get_dummies() to be able to use categorical data in the analysis
temp = pd.get_dummies(Hate_Crimes['Bias Motive Description'])

#This column will determine if a Hate Crime is Anti-Asian or not
Hate_Crimes['ANTI-ASIAN'] = temp['ANTI-ASIAN']


#Split the data to have a training set for the model and a testing set

X_train, X_test, y_train, y_test = train_test_split(Hate_Crimes['Month Number'].to_numpy(),Hate_Crimes['ANTI-ASIAN'],test_size=0.4,random_state=42)

#Run the model on the training data

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

clf = LogisticRegression()
clf.fit(X_train, y_train)
```

Now that the model has been trained, it is important to see how well this model performs through getting its score and setting up a confusion matrix.


```markdown
#Obtain the score of the model using the testing set
score = clf.score(X_test,y_test) 

y_predict = clf.predict(X_test)

#Building the confusion matrix
confuse_mx = metrics.confusion_matrix(y_test,y_predict,labels=[1,0])

#Graph the confusion matrix using a heatmap
bad_logisitic_model =sns.heatmap(confuse_mx, square=True, annot=True, fmt='d', 
                                 cbar=True,linewidths=0.2,
                                 xticklabels=['Anti-Asian', "Not Anti-Asian"],
                                 yticklabels=['Anti-Asian', "Not Anti-Asian"],
                                )

plt.xlabel('True Label',fontsize=18)
plt.ylabel("Prediced Label",fontsize=18)
plt.title("Score: 0.87", fontsize=14)
plt.suptitle("Whether a General Hate Crime is Anti-Asian", fontsize=20)

plt.savefig('WhetherCrimeAntiAsian.png', bbox_inches='tight')

plt.show()
```
![WhetherCrimeAntiAsian](/Crimes-Before-During-Pandemic/assets/css/WhetherCrimeAntiAsian.png)

This confusion matrix showcases that while the model does have an impressive score of 0.87, it never predicted whether the crime was Anti-Asian correctly. Therefore, it raises suspision that the only reason it scored so high, was because the majority of cases weren't Anti-Asian in the dataset. As a result, the model was going to be correct most of the time. 

Now it is best to represent the data and the logistic regression model's predicted probability through a lmplot, where 0 would be when the case isn't Anti-Asian and 1 it is. There would a line through the graph, showcasing the model's determined probability of the case being Anti-Asian, depending on the month the case occured. 

```markdown
#Obtain the columns necessary
data = Hate_Crimes[['ANTI-ASIAN']]
data['Month Number'] = Hate_Crimes['Month Number']

#Create a different jitter function to remove overlap for this data
def jitter_df2(df, x_col, y_col):
    x_jittered = df[x_col] + np.random.normal(scale=0, size=len(df))
    y_jittered = df[y_col] + np.random.normal(scale=0.05, size=len(df))
    return df.assign(**{x_col: x_jittered, y_col: y_jittered})

#Create the graph with the columns
bad_lm_pred = sns.lmplot(x='Month Number', y='ANTI-ASIAN',
           data=jitter_df2(data,'Month Number','ANTI-ASIAN'),
           fit_reg=False)

#This will be the X values used to predict
xs = np.linspace(-1, 13, 100)
#Predict the Y's
ys = clf.predict_proba(xs.reshape(-1, 1))[:, 1]

#Plot the line
plt.plot(xs, ys)
plt.title("Hate Crime Data and Predicted Probability")
plt.savefig('HateCrimeLogisticModel1.png', bbox_inches='tight')
plt.show()          
```

![HateCrimeLogisticModel1](/Crimes-Before-During-Pandemic/assets/css/HateCrimeLogisticModel1.png)

This graph then showcases that no matter the month, the model is going to predict that the case isn't Anti-Asian. As a result, this shows that the month cannot be an indicator of whether a Hate Crime is Anti-Asian. 

### Second Attempt at creating a Logistic Model 

This time, the goal is to investigate whether a ***race-related*** Hate Crime is Anti-Asian. Using similar code, the two following graphs are generated:


![WhetherRaceCrimeAntiAsian](/Crimes-Before-During-Pandemic/assets/css/WhetherRaceCrimeAntiAsian.png)


![HateCrimeLogisticModel2](/Crimes-Before-During-Pandemic/assets/css/HateCrimeLogisticModel2.png)

Unlike before, the confusion matrix here is showing that the model is capable of predicting a case is Anti-Asian correctly, by using the month the case occured. However, unlike before, the accuracy of the model has suffered by having a 0.52 score. 

Additionally, the second graph showcases the model is a lot more capable of predicting a case as Anti-Asian. After all, it shows that there is more than a 50% chance a race related hate crime is Anti-Asian if it occurred between January-March. Yet, this entire model can be disregarded because of its poor score. 

Therefore, once again, the month a case occurred cannot be used to predict whether a race motivated crime is Anti-Asian. 

## How were all crimes affected by the Pandemic?

Unlike before, where the data so far was focused on Hate Crimes, this time the data encompasses all types of crimes. The goal now is to see how these cases are distributed amongst different groups of people.

Before investigations begin, it is important to clean the data first. Using the two datasets: ____ and ______, the code below gets the information ready. 

```markdown

# Fix the last column to match name, they're the same but named differently
ArrestDataHistoric = ArrestDataHistoric.rename(columns={"Lon_Lat": "New Georeferenced Column"})


#Convert the arrest date to datetime  
ArrestDataHistoric['ARREST_DATE'] = pd.to_datetime(ArrestDataHistoric['ARREST_DATE'])

ArrestDataCurrent['ARREST_DATE'] = pd.to_datetime(ArrestDataCurrent['ARREST_DATE'])


#Obtain the data from 03/11/2018 - 09/30/2019, when the pandemic wasn't a thing yet. 
Arrest_Pre_Pandemic = ArrestDataHistoric.loc[(ArrestDataHistoric['ARREST_DATE'] >= "03/11/2018")
                                     & (ArrestDataHistoric['ARREST_DATE'] <= '09/30/2019')]


#Grab the dates from start of the pandemic (2020) within Historic
#ArrestDataCurrent doesn't contain the dates of the pandemic within 2020
#Dates added will be from 03/11/2020 - 12/31/2020

start_of_pandemic = "03/11/2020"

Arrest_Pandemic_Start = ArrestDataHistoric.loc[(ArrestDataHistoric['ARREST_DATE'] >= start_of_pandemic)]

# so take the 2020 year arrest data and add those rows to the current arrest data
Pandemic_Arrests  = ArrestDataCurrent.append(Arrest_Pandemic_Start)

#Pandemic_Arrests['Year'] = Pandemic_Arrests['ARREST_DATE'].dt.year

```

Now there are two dataframes, where one hold all the arrests between 03/11/2018 - 09/30/2019, and the other holding cases from 03/11/2020 - 09/30/2020


### Differences Amongst Gender

After obtaining data showcasing the amount of cases per borough, seperated by gender for both before the pandemic and during, the goal is to showcase the data in 4 seperate columns on a bar graph. 

```markdown

#Obtain the columns containing the amount of cases per borough where the perpretrator was male or female
boroughGender = boroughPrePandemic[['Perp_Female', 'Perp_Male']]
#Rename the columns to showcase it is data from before the pandemic
boroughGender = boroughGender.rename(columns={"Perp_Female": "Female (Pre-Pandemic)",
                                              "Perp_Male": "Male (Pre-Pandemic)" })
                                              
#Obtain the data for during the pandemic
boroughGender['Female (Pandemic)'] = boroughPandemic['Perp_Female']
boroughGender['Male (Pandemic)'] = boroughPandemic['Perp_Male']

#Sneaky way to get all the columns, but no rows
genSum = boroughGender[0:0]
genSum=genSum.reset_index(drop=True)

#Obtain the sum of each column
colsum = boroughGender.sum(axis=0)

#Add the sums as a row in genSum
genSum =genSum.append(colsum, ignore_index= True)

borogen_plt = sns.barplot(data=genSum, 
                          order=["Female (Pre-Pandemic)", "Female (Pandemic)",
                                 'Male (Pre-Pandemic)', 'Male (Pandemic)'])
plt.suptitle('Crimes Committed Before vs. During Pandemic',fontsize=20)
plt.title("Seperated by Gender",fontsize=18)
plt.ylabel('Number of Cases', fontsize=15)
borogen_plt.set_xticklabels(borogen_plt.get_xticklabels(),
                            rotation=30, ha="right")

plt.savefig('GenderBeforeDuring.png', bbox_inches='tight')
plt.show()

```

![GenderBeforeDuring](/Crimes-Before-During-Pandemic/assets/css/GenderBeforeDuring.png)

As a result, the data showcases that there was a decrease in the amount of crimes overall for both genders during the pandemic. This could be because the Pandemic caused less people to travel outdoors, thus less accidents. Meanwhile, males clearly performed more crimes than females overall. 

### Gender, Borough, and Number of Crimes

It is important to visualize how the amount of crimes differed between before and during the pandemic. For that reason, the data is going to be grouped in two ways: By borough and Gender. This way, it will further showcase the differences between Sex while highlighting the differences between the boroughs. 

Firstly, the code below will showcase how the data before the pandemic looks. 

```markdown
#Group the data
preboro = Arrest_Pre_Pandemic.groupby(["ARREST_BORO","PERP_SEX"]).count()
preboro = preboro.reset_index()

#Plot the data with a barplot
#y is ARREST_KEY since the groupby contained count(), thus highlighting how many cases there were in each group

borogen_plt = sns.barplot(x="ARREST_BORO", y ="ARREST_KEY", hue = "PERP_SEX",data=preboro,
                          palette=['indianred','goldenrod'])

plt.suptitle('Crimes Pre-Pandemic (03/11/2018 - 09/30/2019)', fontsize=20)

plt.xlabel('Borough', fontsize=17)
plt.ylabel('Number of Cases', fontsize=15)

borogen_plt.set_xticklabels(["Bronx", "Brooklyn", "Manhattan","Queens","Staten Island"])
legend = borogen_plt.get_legend()
legend.set_title("Perpetrator Sex")
plt.savefig('CrimesPrePandemic.png', bbox_inches='tight')
plt.show()

```

![Crimes Before Pandemic](/Crimes-Before-During-Pandemic/assets/css/CrimesPrePandemic.png)

This graph reveals that in each borough, males perform a lot more crimes. Furthermore, Brooklyn faced the most crimes, while Staten Isalnd faced the least. It is worth noting that women never exceeded 20000 crimes, while in Brooklyn males performed over 80000. 

Now the code will show how this changed during the Pandemic. 

```markdown

#Group the data
during_boro = Pandemic_Arrests.groupby(["ARREST_BORO","PERP_SEX"]).count()
during_boro = during_boro.reset_index()

#Create the Graph
borogen_plt = sns.barplot(x="ARREST_BORO", y ="ARREST_KEY", hue = "PERP_SEX",data=during_boro,
                          palette=['Red','darkslategrey'])

#Edit the graph
plt.suptitle('Crimes During Pandemic (03/11/2020 - 09/30/2021)', fontsize=20)

plt.xlabel('Borough', fontsize=17)
plt.ylabel('Number of Cases', fontsize=15)

borogen_plt.set_xticklabels(["Bronx", "Brooklyn", "Manhattan","Queens","Staten Island"])
legend = borogen_plt.get_legend()
legend.set_title("Perpetrator Sex")
plt.savefig('CrimesDuringPandemic.png', bbox_inches='tight')
plt.show()
 ```

![Crimes During Pandemic](/Crimes-Before-During-Pandemic/assets/css/CrimesDuringPandemic.png)

Interestingly, at first glance the pattern of the graph looks near identical to the previous. Like before, Brooklyn contains the most cases, while Staten Island faced the least. However, it is worth paying attention to the scale. The scale showcases the crimes within each borough decreased significantly. 

To highlight the differences, the code below will overlap the two graphs. The goal was to plot both graphs onto the same figure. 

```markdown

#This is near identical to the first graph's code:

preboro = Arrest_Pre_Pandemic.groupby(["ARREST_BORO","PERP_SEX"]).count()
preboro = preboro.reset_index()
borogen_plt = sns.barplot(x="ARREST_BORO", y ="ARREST_KEY", hue = "PERP_SEX",data=preboro,
                          palette=['indianred','goldenrod'])

plt.suptitle('Crimes Pre-Pandemic (03/11/2018 - 09/30/2019)', fontsize=20)

plt.xlabel('Borough', fontsize=17)
plt.ylabel('Number of Cases', fontsize=15)

borogen_plt.set_xticklabels(["Bronx", "Brooklyn", "Manhattan","Queens","Staten Island"])



#This is near identical to the second graph's code:

during_boro = Pandemic_Arrests.groupby(["ARREST_BORO","PERP_SEX"]).count()
during_boro = during_boro.reset_index()
borogen_plt = sns.barplot(x="ARREST_BORO", y ="ARREST_KEY", hue = "PERP_SEX",data=during_boro,
                          palette=['Red','darkslategrey'])

plt.suptitle('Crimes Pre/During Pandemic (03/11/2018 - 09/30/2021)', fontsize=20)

plt.xlabel('Borough', fontsize=17)
plt.ylabel('Number of Cases', fontsize=15)

borogen_plt.set_xticklabels(["Bronx", "Brooklyn", "Manhattan","Queens","Staten Island"])

#Needed to alter the legend to make sure it is labeled correctly for each bar

plt.legend(title='Perpetrator Sex', loc='upper right', labels=['F Pre-Pandemic', 'M Pre-Pandemic', 
                                                               'F Pandemic', 'M Pandemic'],
           bbox_to_anchor= (1.25, 1) )

#Set the colors of the legend so it matches with the colors of the bars
legend = borogen_plt.get_legend()
legend.legendHandles[0].set_color('indianred')
legend.legendHandles[1].set_color('goldenrod')
legend.legendHandles[2].set_color('red')
legend.legendHandles[3].set_color('darkslategrey')

plt.savefig('CrimesPreDuringPandemic.png', bbox_inches='tight')
plt.show()
```

![Crimes Before and During Pandemic](/Crimes-Before-During-Pandemic/assets/css/CrimesPreDuringPandemic.png)

This graph then clarifies the huge differences in the amount of crimes before and during the pandemic. While the pattern is similar, the crimes faced during the Pandemic is cut almost half it was before. 

### Highlighting How Many Crimes Were Committed By A Member of Each Race

![Pie1](/Crimes-Before-During-Pandemic/assets/css/Pie1.png)

![Pie2](/Crimes-Before-During-Pandemic/assets/css/Pie2.png)



