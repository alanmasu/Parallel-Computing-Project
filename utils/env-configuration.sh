module load Python
python3 -m venv graphEnv
source graphEnv/bin/activate
export PYTHONPATH=$PWD/graphEnv/lib/python3.8/site-packages:$PYTHONPATH
# python3 -m pip install -U pip
python3 -m pip install -U matplotlib
python3 -m pip install seaborn pandas
python3 utils/python_env_exclusion.py
deactivate