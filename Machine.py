import pandas as pd
import numpy as np
class Machine(object):
    def __init__(self):
        self.machine_set=None
        self.id=None
        self.type_id=None
        self.start_time=None
        self.end_time=None

    def load_machine(self,file_id,table_id):
        table_name="./data/machine-{}.xlsx".format(file_id)
        sheet_name="Sheet{}".format(table_id)
        data=pd.ExcelFile(table_name)
        self.machine_set=data.parse(sheet_name,index_col="机器编号")
        self.machine_set=np.array(self.machine_set,dtype='int64')
        return self.machine_set