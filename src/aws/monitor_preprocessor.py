"""
Model Monitor record preprocessor, uploaded to S3 by `aws-tools endpoint --up`.

The endpoint receives {"instances": [...]}; Model Monitor needs one flat record
per captured request, so the first customer of each request is checked against
the training baseline.
"""

import json


def preprocess_handler(inference_record):
    return json.loads(inference_record.endpoint_input.data)['instances'][0]
