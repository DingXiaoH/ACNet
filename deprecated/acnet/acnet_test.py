from ding_test import general_test
from acnet.acnet_fusion import convert_acnet_weights
from acnet.acnet_builder import ACNetBuilder
import sys

def convert_and_test(network_type, train_weights):
    builder = ACNetBuilder(base_config=None, deploy=False)
    general_test(network_type=network_type, weights=train_weights, builder=builder)
    deploy_weights = train_weights.replace('.hdf5', '_deploy.hdf5')
    convert_acnet_weights(train_weights, deploy_weights=deploy_weights, eps=1e-5)
    builder.switch_to_deploy()
    general_test(network_type=network_type, weights=deploy_weights, builder=builder)

if __name__ == '__main__':
    convert_and_test(sys.argv[1], sys.argv[2])

