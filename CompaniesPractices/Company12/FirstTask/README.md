### 1st Task Solution
Before proceed with this task, main thing we should be aware in - which environment we are going to setup and work with. 
E.g.:
- we have main local server where all setup will be define, such as:
    - environment for executing jobs {python + external libs}
    - database to store data (with accessible UI from external) {SQL/NoSQL/Warehouse/DataLake}{MongoDB, OracleDB, ...}
    - database engine to curate data (with accessible UI from external) {Workbench, Dremio, Drill, Compass ...}
    - scheduler to exucute jobs (with accessible UI from external) {Rundeck, Airflow, Luigi, CronTab ...}
    - place to store all codebase (with accessible UI from external) {GitHub, GitLab, BitBucket ...}
    - CI/CD (with accessible UI from external) {Atlassian Bamboo, Jenkinks, ...}
    - Dockerizing - do we need to wrap and run solutions within Docker? Nutshell? 

- we DO NOT have main local server, instead all operations will be setup and executed within Cloud Servers:
    - environment for executing jobs {AWS EC2 + AWS Lambda functions}
    - database to store data (with accessible UI from external) {S3, Redshift, Aurora} {(DataBases)[https://aws.amazon.com/products/databases/], (Storages)[https://aws.amazon.com/products/storage/]}
    - database engine to curate data (with accessible UI from external) {AWS Console}
    - scheduler to exucute jobs (with accessible UI from external) {AWS CloudWatch}
    - place to store all codebase (with accessible UI from external) {GitHub, GitLab, BitBucket ...}
    - CI/CD (with accessible UI from external) {AWS CI/CD}

P.S. The stack of components should be setup with the companie's and team's requirements and abilities. (especially finance abilities).    

#### Straight-forward approach <- (additinal ERD)[https://miro.com/app/board/o9J_kqOYPRc=/]
- Assume we have multiple external sources to obtain data from. As well we have domain knowledge about these external sources, i.e. we (roughtly) know what data types and format we expect. 

##### Pipeline description
    - Firstly, we could classify external resources by their types.
    - Per each type, we could created chain of custom, but reusable scripts for ETL:
        - scrap_data.py
        - extract_data.py
        - transform_data.py
    - Plus, we could create scripts whcih are independent of scrap/extract/transform steps, because we expect just one format after transformation step:
        - training_db_load_data.py
        - sync_data.py (should be triggered after data was loaded to training_db)
        - production_db_load_data.py

##### Tools description
`Scraping`
- requests (static web applicatins)
- scrapy <- can be used here
- urlib <- can be used here

`Extracting HTML`
- codec
- selenium (dynamic web applicatins) {could setup Automated web scraping}
- bs4 (static web applicatins)
- lxml (static web applicatins)
- scrapy <- can be used here (By the processes definitions, better to use on this step)
- urlib <- can be used here (By the processes definitions, better to use on this step)


`Extracting CSV`
- pandas

`Extracting PDF`
- PyPDF2
- textract

`Transforming`
- pandas 
- json
- os
- pathlib
- ... pure python libriaries


`Loading`
- ODBC {Depends on DB type}
- custom write/loader OR imported write/loader
- boto3 {for AWS services}

#### "Costly" approach <- (additinal ERD could be used also for high-level overview)[https://miro.com/app/board/o9J_kqOYPRc=/]
- Assume we have multiple external sources to obtain data from. As well we have domain knowledge about these external sources, i.e. we (roughtly) know what data types and format we expect.
- We could use completely cloud solution within AWS Servicies:
    
    - AWS API GateWay
    - AWS Lambda
    - AWS Textract

    - AWS S3
    - AWS RDBS or other DB types
    
    - AWS CloudWatch
    
    - AWS Cloud9
    - AWS EC2
    - AWS Data PipeLine
    - AWS Cloudformation or Serverless

AWS services allow define and create very flexible pipeline(S) such as you wish. All depends on specialist knowledge, expirience as well as companie's requirements and timeboxes.
   
    
### Conclusion
First of all, of course, before suggesting something it would be vital to explore and obtain domain knowledge about data.
Afterward, it would be possible to make few experiments with different combinations of approaches.
As well, it would be neccessary to develop `data quality check` step which will allow undersatnd your's data statistics. Moreover, it would be beneficial for company's head-quater to see different types of visualisation. E.g.: this month we proceeses 1500 companies. 700 companies have good mark, 300 companies have medium mark, 500 companies have low mark and marketing department should pay attention on them and invoke additional inspections.       


