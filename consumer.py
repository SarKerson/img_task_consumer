import json
import io
import redis
import logging


from kafka import KafkaConsumer
from google.cloud import storage

from test import inference


HOST = '35.240.132.243'
BUCKET_NAME = 'ylq_server'
URL = 'https://storage.googleapis.com/ylq_server/%s'

# redis
rc = redis.Redis(host=HOST, port=6379, db=0)
# kafka
consumer = KafkaConsumer(
    'gan',
    bootstrap_servers='%s:9092' % HOST,
)
# cloud storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)


import os
ROOT = os.path.dirname(os.path.abspath(__file__))
logfile = "{0}/logs/app.log.{1}".format(ROOT, time.strftime("%Y%m%d"))
fh = logging.FileHandler(filename=logfile)
fmt = logging.Formatter(fmt="%(asctime)s|%(levelname)s|%(process)d|%(message)s", datefmt="%Y/%m/%d %H:%M:%S")
fh.setFormatter(fmt)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(fh)


class TaskMeta(object):
    """
    	TaskID     string `json:"task_id"`
	UserID     string `json:"user_id"`
	CreateTime int64  `json:"create_time"`
	ProcStatus int64  `json:"proc_status"`
	InputURL   string `json:"input_url"`
	OutputURL  string `json:"output_url"`
    """

    KEY = 'test:%s'

    TASK_STATUS_PROCESSING = 1
    TASK_STATUS_SUC        = 2
    TASK_STATUS_FAILED     = 10

    def __init__(self, task_id, user_id, create_time, proc_status, input_url, output_url):
        self.task_id = task_id
        self.user_id = user_id
        self.create_time = create_time
        self.proc_status = proc_status
        self.input_url = input_url
        self.output_url = output_url

    @classmethod
    def from_json(cls, d):
        return cls(
            d['task_id'],
            d['user_id'],
            d['create_time'],
            d['proc_status'],
            d['input_url'],
            d['output_url'],
        )

    @classmethod
    def from_str(cls, s):
        d = json.loads(s)
        return cls.from_json(d)

    @classmethod
    def from_redis(cls, task_id):
        try:
            return cls.from_str(rc.get(cls.KEY % task_id))
        except Exception as why:
            return None

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.to_dict())

    def update(self):
        try:
            rc.set(self.KEY % self.task_id, self.to_json())
            return True
        except Exception as why:
            return False


def upload_img(task_id, img):
    try:
        obj_name = '%s_res' % task_id
        b = io.BytesIO()
        img.save(b, 'PNG')
        bucket.blob(obj_name).upload_from_string(b.getvalue())
        return obj_name
    except:
        return None


ERR_READ_REDIS_FAILED = 1
ERR_WRITE_REDIS_FAILED = 2
ERR_WRITE_CLOUD_STORAGE_FAILED = 3
SUC = 0


def process(task_id):
    # get meta info
    meta = TaskMeta.from_redis(task_id)
    logging.info('[process] msg: %s', task_id)
    if not meta:
        logging.error('[process] get from_redis failed, task_id: %s', task_id)
        return ERR_READ_REDIS_FAILED

    # inference
    result_img = inference(meta.input_url)

    # upload result
    obj_name = upload_img(meta.task_id, result_img)
    logging.info('[process] upload_img: %s, obj_name: %s', task_id, obj_name)
    if not obj_name:
        logging.error('[process] upload_img failed, task_id: %s', task_id)
        return ERR_WRITE_CLOUD_STORAGE_FAILED
    output_url = URL % obj_name

    # update meta info
    meta.output_url = output_url
    meta.proc_status = TaskMeta.TASK_STATUS_SUC
    logging.info('[process] meta to update: %s', meta.to_dict())
    if not meta.update():
        logging.error('[process] meta.update failed, task_id: %s', task_id)
        return ERR_WRITE_REDIS_FAILED
    logging.info('[process] done task: %s', task_id)
    return SUC


def run():
    for msg in consumer:
        if msg.key != 'commit':
            continue
        logging.info('[run] msg: %s', msg.value)
        r = process(msg.value)
        logging.info('[run] done msg: %s', r)


def test():
    input_url = 'https://storage.googleapis.com/ylq_server/2a391a22368eeabbcfdd7771e99a8ca3'
    task_id = '2a391a22368eeabbcfdd7771e99a8ca3'
    result_img = inference(input_url)
    # upload result
    obj_name = upload_img(task_id, result_img)
    if not obj_name:
        return 1
    output_url = URL % obj_name
    print(output_url)


if __name__ == '__main__':
    test()
