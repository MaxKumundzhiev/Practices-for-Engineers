# Forward Deployed Engineer challenge
In this exercise, we'll execute a data extraction from an imaginary data partner. 

# Solution Report
:construction: Solution was provided by Maksim Kumundzhiev. :construction:

# Setup Environment
We assume you already have installed conda package manager.
If not, follow [Anaconda official documentation](https://docs.anaconda.com/anaconda/install/).

- Create dedicated environment 
```bash
$ conda create -n kheiron-test' python=3.7
```

- Activate created environment
 ```bash
$ conda activate kheiron-test 
```

- Install necessary requirements
 ```bash
$ pip install -r requirements.txt 
``` 

# Launch
Get help:
```bash
python -m main.handler -h
```

Run main script and obtain results:
1. replace `<your_path>/...` with related path to files
2. execute following command:

```bash
$ python -m main.handler --lim <your_path>/lims.txt --ris <your_path>/ris.csv --pac <your_path>/pacs.json.csv --save <your_path>/imaginary_partner_patients.txt
```


#### Sample output
```bash
{"patient_uid": "MS4yLjgyNi4wLjEuMTAxODQxMDMuNS4xLjgwNzUwS0hFSVJPTl9URVNUX1RBU0s=", "sex": "F", "date_of_birth": "1962.7.25", "studies": [{"StudyID": "S0hFSVJPTl9URVNUX1RBU0s=", "StudyDate": "2022.11.4", "StudyTime": "000000.000000", "StudyDescription": NaN, "StudyInstanceUID": "MS4yLjgyNi4wLjEuMTAxODQxMDMuNS40LjE3NTMwMktIRUlST05fVEVTVF9UQVNL", "StudyStatusID": "Q09NUExFVEVES0hFSVJPTl9URVNUX1RBU0s=", "StudyComments": NaN}, {"StudyID": "S0hFSVJPTl9URVNUX1RBU0s=", "StudyDate": "2022.11.2", "StudyTime": "000000.000000", "StudyDescription": NaN, "StudyInstanceUID": "MS4yLjgyNi4wLjEuMTAxODQxMDMuNS40LjE3NTMwMktIRUlST05fVEVTVF9UQVNL", "StudyStatusID": "Q09NUExFVEVES0hFSVJPTl9URVNUX1RBU0s=", "StudyComments": NaN}, {"StudyID": "S0hFSVJPTl9URVNUX1RBU0s=", "StudyDate": "2022.11.4", "StudyTime": "000000.000000", "StudyDescription": NaN, "StudyInstanceUID": "MS4yLjgyNi4wLjEuMTAxODQxMDMuNS40LjE3NTMwMktIRUlST05fVEVTVF9UQVNL", "StudyStatusID": "Q09NUExFVEVES0hFSVJPTl9URVNUX1RBU0s=", "StudyComments": NaN}, {"StudyID": "S0hFSVJPTl9URVNUX1RBU0s=", "StudyDate": "2022.11.4", "StudyTime": "000000.000000", "StudyDescription": NaN, "StudyInstanceUID": "MS4yLjgyNi4wLjEuMTAxODQxMDMuNS40LjE3NTMwMktIRUlST05fVEVTVF9UQVNL", "StudyStatusID": "Q09NUExFVEVES0hFSVJPTl9URVNUX1RBU0s=", "StudyComments": NaN}], "rad": {"side": ["R2", "R2"], "date": "2014.8.12", "opinion": "R2"}, "patho": {"date": NaN, "opinion": NaN}}
``` 

# Explonatory and Consistency Issues
### PAC 
PAC table primary key: `accession number`

PAC table maps with RIS table with 'accession_number_id' <--> 'id'  

- Total number of `records == 904` <- it means there were made 904 observations
- Amount of `all patient ids == 904`
- Amount of `unique patient ids == 290` <- it means there were repetitive patients treatments    

- Amount of `all accession numbers` == 904
- Amount of `unique accession number` == 248

####Consistency Issues:
```
1.
- Amount of `unique patient ids == 290`
    - out of patient ids found `71 '"incorrectly" formed'` 
        e.g.: usual form: `1.2.826.0.1.10184103.5.1.106101` vs found form: `df70ee3f-8f64-4ab4-af78-a3a739a0a347`
 
As for `"incorrectly" formed` - my assumption is that it is not "wrong" form. These ids just were obtained from other system.
My approach was firstly keep them, because they did not have impact for mapping data. 
Afterwards, I applied same encoding for them as for usual form as `1.2.826.0.1.10184103.5.1.106101`.
P.S. 
As well, I tried to decode such ids, but decided not to spend a lot time for that.
To correctly handle this clause need further investigation/consultation of function which generate these IDs.

2.
- Found oddly formed/generated patient's date of birth. 
By provided information all the patients were born at 1888-..-..
I had checked patient's date of birth within RIS data, where dates seemed more realistic. 
Afterwards, I decided to use dates from RIS data.
P.S.
To correctly handle this clause need further investigation/consultation of way of obtaining these dates.            
```    

### RIS
RIS table primary key: `id`
RIS table maps with PAC table with 'id' <--> 'accession_number_id'  
 
- Total number of `records == 248` <- it means it was made 248 observations
- Amount of `all patient ids == 248`
- Amount of `unique patient ids == 80` <- it means there were repetitive patients treatments
- Amount of `all treatment ids == 248`
- Amount of `unique treatment ids == 248`
 


### LIM
LIM table primary key: `id` {from PID, "external_patient_id", see below}
LIM table maps with RIS table with 'id' <--> 'id'

- Total number of `records == 6` <- it means it was made 6 observations
- Amount of `all patient ids == 6`
- Amount of `unique patient ids == 6` <- it means there will be patients who DO NOT have information from LIM

 ####Consistency Issues:
```
- For patient {id: 54e4035a-b9eb-4118-ba1e-635e3f934c8a}, there was no provided "OBR id".
Filled with np.nan, then encoded.  

- All ids from PID after extracting were provided formed as: MVZb4dcdc27-b876-4ee1-a24d-3d14472f96ee
After investigations, it was realised, that first 3 characters {MVZ} could be replaced. 
Assumed following, I created 2 keys, named: "external_patient_id" {droped 3 characters} and "internal_patient_id" {kept original}.     

If skip this step, we can assume that we can map with RIS table using "OBX id", but:
- found one missing "OBX id", what will invoke missing data in general
- if make comparison and check amount of records you obtain using mapping with "OBX id" and "PID id" as:
##############################
ris_df[ris_df['id'] == 'e27e13cd-3c1e-4816-b810-d11a6103224d'] # by observation id
ris_df[ris_df['pat_id'] == 'b4dcdc27-b876-4ee1-a24d-3d14472f96ee'] # by patient id      
##############################
you will realise, that you will lose resulting data, because observation IDS in RIS table are UNIQUE. 

P.S.
To correctly handle this clause need further investigation/consultation of reason of missing value.
```

## IAM
**[Malsim Kumundzhiev](https://github.com/KumundzhievMaxim)**

[<img src="http://i.imgur.com/0o48UoR.png" width="35">](https://github.com/KumundzhievMaxim)             [<img src="https://i.imgur.com/0IdggSZ.png" width="35">](https://www.linkedin.com/in/maksim-kumundzhiev/)             [<img src="https://loading.io/s/icon/vzeour.svg" width="35">](https://www.kaggle.com/maximkumundzhiev) 