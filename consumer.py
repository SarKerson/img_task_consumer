import json
import io
import redis

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
        return cls.from_str(rc.get(cls.KEY % task_id))

    def to_dict(self):
        return self.__dict__

    def to_json(self):
        return json.dumps(self.to_dict())

    def update(self):
        rc.set(self.KEY % self.task_id, self.to_json())


def upload_img(task_id, img):
    try:
        obj_name = '%s_res' % task_id
        b = io.BytesIO()
        img.save(b, 'PNG')
        bucket.blob(obj_name).upload_from_file(b)
        return obj_name
    except:
        return None


def process(task_id):
    meta = TaskMeta.from_redis(task_id)
    # inference
    result_img = inference(meta.input_url)
    # upload result
    obj_name = upload_img(meta.task_id, result_img)
    if not obj_name:
        return 1
    output_url = URL % obj_name
    # update meta
    meta.output_url = output_url
    meta.proc_status = TaskMeta.TASK_STATUS_SUC
    meta.update()


def run():
    for msg in consumer:
        if msg.key != 'commit':
            continue
        process(msg.value)


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
