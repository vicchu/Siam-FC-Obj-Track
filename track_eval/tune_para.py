import os
import sys
import logging
from gaft import GAEngine
from gaft.components import BinaryIndividual, Population
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from gaft.analysis import FitnessStore
from multiprocessing import Value

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from track_eval.test_net import *

tracker_num = Value('i', 0)
if __name__ == '__main__':
    arg_set.logger_handler.removeHandler(arg_set.logfile_handler)
    indv_template = BinaryIndividual(ranges=[(1.0000001, 1.25), (0.9, 1.0), (0.15, 0.7), (0.05, 0.7)], eps=0.0001)
    population = Population(indv_template=indv_template, size=50)
    population.init()
    selection = RouletteWheelSelection()
    crossover = UniformCrossover(pc=0.8, pe=0.5)
    mutation = FlipBitMutation(pm=0.1)
    engine = GAEngine(population=population,
                      selection=selection,
                      crossover=crossover,
                      mutation=mutation,
                      analysis=[FitnessStore])


    # @engine.analysis_register
    # class ConsoleOutput(OnTheFlyAnalysis):
    #     master_only = True
    #     interval = 1
    #
    #     def register_step(self, g, population, engine):
    #         best_indv = population.best_indv(engine.fitness)
    #         scale_step, scale_penalty, scale_lr, hann_weight = best_indv.solution
    #         msg = 'Generation: {}, ACC:{:.3f}, step:{:.4f}, penalty:{:.4f}, lr:{:.4f}, hann:{:.4f}' \
    #             .format(g, engine.fmax, scale_step, scale_penalty, scale_lr, hann_weight)
    #         # engine.logger.info(msg)
    #         # print(msg)

    @engine.fitness_register
    def fitness(indv):
        scale_step, scale_penalty, scale_lr, hann_weight = indv.solution
        with tracker_num.get_lock():
            tracker_num.value += 1
            tracker_name = 'SFC_TUNE_' + arg_set.time_str + '_{:02d}'.format(tracker_num.value)
        # return scale_step + scale_penalty + scale_lr + hann_weight
        tracker_tune = SFC(tracker_name, indv.solution)
        exp = ExperimentOTB(arg_test.data_root,
                            version=2013,
                            result_dir=arg_test.result_dir,
                            report_dir=arg_test.report_dir)
        exp.run(tracker_tune)
        performance = exp.report([tracker_tune.name])
        performance = float(performance[tracker_name]['overall']['success_score'])
        msg = 'ACC:{:.4f}, step:{:.4f}, penalty:{:.4f}, lr:{:.4f}, hann:{:.4f}' \
            .format(performance, scale_step, scale_penalty, scale_lr, hann_weight)
        print(msg)
        return performance


    engine.run(ng=100)
