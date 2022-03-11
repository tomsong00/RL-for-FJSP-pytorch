#编号	所属大任务	内部编号	允许执行的机器类型编号	准备时间	加工时间
import pandas as pd
import numpy as np
class Task(object):
    def __init__(self):
        self.task_set=None
        self.id=None
        self.job_id=None
        self.interal_id=None
        self.permit_id=None
        self.perpare_time=None
        self.process_time=None

    def load_task(self,file_id,table_id):
        table_name="./data/task-{}.xlsx".format(file_id)
        sheet_name="Sheet{}".format(table_id)
        data=pd.ExcelFile(table_name)
        self.task_set=data.parse(sheet_name,index_col="编号")
        self.task_set=np.array(self.task_set,dtype='int64')
        return self.task_set