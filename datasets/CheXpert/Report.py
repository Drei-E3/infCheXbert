import pandas as pd

class Report:

    def __init__(self,file,patient_id= None,study_id= None):
        self.file = file
        self.patient_id = patient_id
        self.study_id = study_id


    def report_parseto_df(self):
        file= open(self.file,'r')
        valve_finding = False
        valve_impression = False
        txt=file.read()
        file.seek(0)
        finding = ''
        impression = ''

        for line in file:
            if 'FINDINGS' in line:
                valve_finding = True
                continue
            if 'IMPRESSION' in line:
                valve_impression = True
                valve_finding = False
                continue
            if valve_finding == True:
                finding = finding +' ' + line 
                finding.strip()    
            if valve_impression == True:
                impression = impression +' ' + line 
                impression.strip()
       
        file.close()
        
        report = pd.DataFrame(
            [[self.patient_id,self.study_id,txt,finding,impression]], 
            columns=['patient_id','study_id','report_txt','finding','impression'])
        
        return report
