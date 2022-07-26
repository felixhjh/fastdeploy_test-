#!/bin/bash
export py_version=python
export MODEL_PATH=/huangjianhui/fastdeploy_test/models
export DATA_PATH=/huangjianhui/fastdeploy_test/data
CURRENT_DIR=$(cd $(dirname $0); pwd)
bash ${CURRENT_DIR}/compile.sh
rm -rf result.txt
cases=`find ./ -name "test*.py" | sort`
echo $cases
ignore=""
bug=0

job_bt=`date '+%Y%m%d%H%M%S'`
echo "============ failed cases =============" >> result.txt
for file in ${cases}
do
    echo ${file}
    if [[ ${ignore} =~ ${file##*/} ]]; then
        echo "跳过"
    else
        if [[ ${ce_name} =~ "cpu" ]]; then
            $py_version -m pytest --disable-warnings -sv ${file} -k "cpu"
        else
            $py_version -m pytest --disable-warnings -sv ${file}
        fi
        if [[ $? -ne 0 && $? -ne 5 ]]; then
            echo ${file} >> result.txt
            bug=`expr ${bug} + 1`
        fi
    fi
done
job_et=`date '+%Y%m%d%H%M%S'`

echo "total bugs: "${bug} >> result.txt
#if [ ${bug} != 0 ]; then
#    cp result.txt ${output_dir}/result_${py_version}.txt
#fi
cat result.txt
cost=$(expr $job_et - $job_bt)
echo "$cost s"
exit ${bug}
