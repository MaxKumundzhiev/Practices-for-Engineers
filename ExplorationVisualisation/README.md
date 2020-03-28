# Course description

The *Data exploration and visualization* course consists of several tutorials and excercise as an introductution into various data retrieval, exploration and visualization techniques. 
There are many databases, which are either publicly available or have limited access and use different services to deal with user queries. One commonly used method is when users can communicate via SQL queries. This is more efficient in case of larger dataqueries, when download time depends on the query script. Other method is to run a REST service that translates user queries to the database. This is easier to use and has it's own advantages.
The retrieved raw data is almost never as clean as we want it to be. It can have unexpected values due to measurement failure or for other reasons. In fact on average 50% effort of data exploration goes into understanding data and its cleaning.

Once data is digested one faces with many options in visualization, which all have their pros and cons. 
In this course we intend to introduce state of the art tools and methods in data exploration and visualization. This field evolves rapidly, like Jupyter notebooks didn't exist some years ago, but some of the programming languages and platforms proved efficient and/or easy to use and became widely used.

We will cover various [packages in python](http://bokae.web.elte.hu/numerical_methods.html) and introduce
[SQL](https://www.w3schools.com/sql/). The types of plotting methods (when and how to use) will be explained and there will be examples of visualization packages 
that can be used in the jupyter notebooks or which create standalone, portable applications. It will be also shown how can we share these plots or how to embed into webpages. 

Each of the assignments will be a combination of theabove mentioned tools. 

The course is held in the North Building in computer lab 5.56 on fridays between 11:15 and 14:00.
There is a one hour lecture followed by a break and two hour laboratory work. Each occasion starts with an introduction to the current topic, like a hands-on session with the prepared notebooks, and then the remaining time is consultation, when lecturers will be available to help with the assignments. 

The schedule is the following:
1.  15.02.2019. **[USGS water discharge statistics](USGS-waterdata-curl-pandas)** - due on 21.03.2019. 
2.  22.02.2019. **[SQL queries on an NBA database](Basketball_League-SQL) I** 
3.  01.03.2019. **[SQL queries on an NBA database](Basketball_League-SQL) II** - due on 28.03.2019.
4.  08.03.2019. **[Following John Snow](John_Snow-geopandas-folium-shapely)** - due on 04.04.2019.
5.  22.03.2019. **[REST services](REST-services)** - due on 25.04.2019.
6.  29.03.2019. **[Visualization](Interactive_Visualization)** 
7.  05.04.2019. **[3D Visualization](3d_Visualization)**
8.  12.04.2019. **[Network exploration](Networkx)** - This lecture will be given by [**Dániel Ábel**](http://maven7.com/hu/daniel-abel/), who is a developer at Maven7.
09. 26.04.2019. **Astrophysics**
10. 03.05.2019. **[Natural Language Processing on tweets](NLP_on_tweets)** - This lecture will be given by **Eszter Bokányi**, whose field of interest is how social phenomena can be captured by using various digital fingerprints of individuals.
11. 17.05.2019. **Consultation**

### Where to work on the assignments?
https://kooplex-edu.elte.hu/hub is where the notebooks will be handed out. It is available for all students and once you run your notebook server you will find a folder with the course material. The notebooks will be available in this Github repository as well.
We will explain how to use this portal on the first lecture.
You can access this portal from outside the university as well e.g. from home. In case there is any problem with the portal you can run a notebook server locally on the lab computers and upload your work later.

### Learning materials
* Python tutorial: http://bokae.web.elte.hu/numerical_methods.html (translated from the BSc course "numerical methods in physics I" by Eszter Bokányi, work in progress )
* SQL tutorial: https://www.w3schools.com/sql/ 
* RESTful service: https://en.wikipedia.org/wiki/Representational_state_transfer
* Networkx: https://networkx.github.io/

### Requirements
There will be an assignment for ech of the 9 topics, which need to be completed individually. The deadlines for the submissions are shown next to the topic and all related information will be in the topics' folder. These are not strict deadlines, however we advise students to keep them in order to be able to complete all tasks.

The **minimum requirement** for this course is to submit all assignments with at least some completed tasks. In each worksheet or assignment the first couple of tasks follow the excercises explained in the given tutorial files.

### Grading

Each assignment get's corrected and a maximum of 10 points can be given. The points will reflect how many worksheets/subtasks are completed,  the quality of the solutions and/or the clarity of the presentation of the work.

### Example Datasets

In the */home/course/public/Dataset* directory you will find datasets, that you can work with.
