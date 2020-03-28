## Statistics of water discharge of natural rivers

In this excercise we will select a few natural rivers and retrieve their data via a REST service. For data retrieval we use either *curl* from command line or *pycurl* from the notebook.

#### What is curl
"cURL is a tool to transfer data from or to a server, using one of the supported protocols (DICT, FILE, FTP, FTPS, GOPHER, HTTP, HTTPS, IMAP, IMAPS, LDAP, LDAPS, POP3, POP3S, RTMP, RTSP, SCP, SFTP, SMB, SMBS, SMTP, SMTPS, TELNET and TFTP). The command is designed to work without user interaction."

USGS water data are available at https://waterdata.usgs.gov/nwis, please visit this site and explore it. It is possible to download data by clicking on the website, but for retrieving large amount of data systematically is cumbersome this way. 
That's why we will learn how to use the datahub's REST service. [REST services](../REST-services) will be covered in detail later.

There is a tutorial notebook to show the basic tools and their functionalities. We load the data into a *pandas* Dataframe and clean it using *pandas* functions.

### Tasks
Complete the tasks in the worksheet.ipynb! You might need to use other tools or packages and there might be multiple acceptable solutions.

### Further Reading

- Probability and statistics for geophysical processes: https://www.itia.ntua.gr/en/docinfo/1322/
- On the probability distribution of daily streamflow in the UnitedStates : https://engineering.tufts.edu/cee/people/vogel/documents/2017_probabilityDistribution.pdf
