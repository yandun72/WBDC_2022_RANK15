FROM tione-wxdsj.tencentcloudcr.com/base/pytorch:py38-torch1.9.0-cu111-trt8.2.5
WORKDIR /opt/ml/wxcode
COPY ./opensource_models ./opensource_models
COPY ./src ./src

COPY ./requirements.txt ./
RUN pip install -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple
#COPY ./*.py ./
COPY ./*.sh ./
COPY ./*.md ./
# CMD sh -c "sh start.sh"
