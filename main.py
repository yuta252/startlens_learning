import logging
from optparse import OptionParser

from app.controllers.training import TrainImage


formatter = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)


def main():
    usage = 'usage: %prog [options] arg1 arg2'
    parser = OptionParser(usage=usage)
    parser.add_option('-m', '--mode', action='store', type='string', dest='mode', help='Set to activate mode')

    options, args = parser.parse_args()

    if options.mode == 'train':
        # TODO: call training model
        print(options.mode)
        print(args)
    elif options.mode == 'knn':
        # TODO: call knn training by spot
        print('KNN train')
    elif options.mode == 'inference':
        # TODO: call KNN infference and activate API server
        print('KNN inference')
    else:
        print("selecet mode by using -m option")


if __name__ == '__main__':
    # main()
    t = TrainImage()
    t.train_model()

