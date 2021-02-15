import logging
from optparse import OptionParser

from app.controllers.training import TrainController
from app.controllers.webserver import start


formatter = '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)


def main():
    usage = 'usage: %prog [options] arg1 arg2'
    parser = OptionParser(usage=usage)
    parser.add_option('-m', '--mode', action='store', type='string', dest='mode', default='server', help='Set to activate mode')

    options, args = parser.parse_args()

    if options.mode == 'train':
        train_controller = TrainController()
        train_controller.train_triplet_model()
    elif options.mode == 'server':
        start()
    else:
        print("selecet mode by using -m option")


if __name__ == '__main__':
    main()
    # test_data = [[
    #     -0.09551838040351868, -0.019040314480662346, -0.04436960071325302, -0.10415035486221313, 0.05173281580209732,
    #     -0.01562654972076416, 0.1402387171983719, -0.10971886664628983, -0.2648703455924988, -0.07008925080299377,
    #     0.0027902848087251186, 0.09803500771522522, 0.17882350087165833, -0.03504638001322746, -0.05005289614200592,
    #     -0.08565837144851685, 0.18177323043346405, 0.017765967175364494, 0.09764382988214493, 0.05077584460377693,
    #     0.06680794060230255, -0.16679325699806213, 0.1814379096031189, 0.08148196339607239, 0.15682151913642883,
    #     -0.30604997277259827, -0.03842161223292351, -0.0673968568444252, -0.21319429576396942, -0.16417427361011505,
    #     0.223836287856102, -0.014128981158137321, 0.03428460285067558, -0.24569539725780487, 0.0027019325643777847,
    #     -0.23574747145175934, 0.021959125995635986, 0.1699928194284439, 0.2961980998516083, -0.033233243972063065,
    #     0.09174466878175735, -0.019155144691467285, 0.050820156931877136, -0.018291544169187546, 0.30083927512168884,
    #     -0.13183045387268066, 0.12874040007591248, 0.24230897426605225, 0.14174704253673553, 0.02702498435974121
    # ]]
