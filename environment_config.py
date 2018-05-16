import socket
from pyspark import SparkContext, SparkConf

"""
Returns objects whose configuration is environment dependent.
"""


def get_cc(hostname):
    """
    Builds an instance of CerebralCortex to suit a known environment.

    Args:
        hostname (str): The hostname of the machine calling for the CerebralCortex instance.

    Returns:
        cc (CerebralCortex): An instance of CerebralCortex configured for the host machine.
    """

    from cerebralcortex.cerebralcortex import CerebralCortex

    if hostname == "cerebralcortex":
        cc = CerebralCortex('/home/vagrant/CerebralCortex-DockerCompose/cc_config_file/cc_vagrant_configuration.yml')

    elif '10dot' in hostname or 'memphis' in hostname:
        cc = CerebralCortex('/cerebralcortex/code/config/cc_starwars_configuration.yml')

    else:
        print("unknownn environment!")
        return None

    return cc

def get_minio_client(hostname):
    """
    Returns an instance of a Minio client configured for the host machine.

    Args:
        hostname (str): The hostname of the machine calling for the Minio client.

    Returns:
        mC (minio.MinioClient): The correctly-configured Minio client.
    """

    from minio import Minio

    if hostname == "cerebralcortex":
        MINIO_IP = '127.0.0.1:9000'
        MINIO_ACCESS_KEY = 'ZngmrLWgbSfZUvgocyeH'
        MINIO_SECRET_KEY = 'IwUnI5w0f5Hf1v2qVwcr'

    elif '10dot' in hostname or 'memphis' in hostname:
        cc = get_cc(hostname)

        MINIO_IP = cc.config['minio']['host'] + ':' + str(cc.config['minio']['port'])
        MINIO_ACCESS_KEY = cc.config['minio']['access_key']
        MINIO_SECRET_KEY = cc.config['minio']['secret_key']

    else:
        print("unknownn environment!")
        return None

    mC = Minio(MINIO_IP,
                  access_key=MINIO_ACCESS_KEY,
                  secret_key=MINIO_SECRET_KEY,
                  secure=False)

    return mC

def get_spark_context(hostname):
    """
    Returns a Spark context configured for the host machine.

    Args:
        hostname (str): The hostname of the machine calling for the Spark context.

    Returns:
        sc (SparkContext): The configured Spark context.
    """

    if hostname == "cerebralcortex":
        sc = SparkContext("local[8]", "MOSAIC")

    elif '10dot' in hostname or 'memphis' in hostname:
        conf = SparkConf().setMaster('spark://dagobah10dot:7077').setAppName('MOSAIC-EMS - ' + sys.argv[1]).set('spark.cores.max','128').set('spark.ui.port','4099').setExecutorEnv('PYTHONPATH',str(os.getcwd()))
        sc = SparkContext(conf=conf)
        sc.addPyFile('/cerebralcortex/code/eggs/MD2K_Cerebral_Cortex-2.2.2-py3.6.egg')

    else:
        print("unknownn environment!")
        return None

    return sc