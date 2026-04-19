import os
import sys
from eval.evaluation import build_foreign_key_map_from_json, evaluate


class Evaluator:
    def __init__(self, table_path, db_dir):
        self.table_path = table_path
        self.db_dir = db_dir
        self.kmaps = build_foreign_key_map_from_json(self.table_path)

    def change_database(self, db_dir):
        self.db_dir = db_dir

    def accuracy(self, pred_filename, dataset, output_path, etype='all'):
        assert etype in ['match', 'exec', 'all']
        result = self.evaluate_with_official_interface(pred_filename, dataset, output_path, etype)
        if etype == 'match':
            return float(result['exact'])
        if etype == 'exec':
            return float(result['exec'])
        return (float(result['exact']), float(result['exec']))

    def evaluate_with_official_interface(self, pred_filename, dataset, output_path, etype='all'):
        gold_filename = os.path.join(os.path.dirname(output_path), 'gold.sql')
        with open(gold_filename, 'w', encoding='utf-8') as tmp_gold:
            for i, example in enumerate(dataset):
                if i > 0:
                    tmp_gold.write('\n')
                for turn in example['interaction']:
                    tmp_gold.write(' '.join(turn['query'].strip('\t ;').split()) + '\t' + example['database_id'] + '\n')
            tmp_gold.flush()
            of = open(output_path, 'w', encoding='utf-8')
            old_print, sys.stdout = sys.stdout, of

            # # TODO print difficulty
            # results = evaluate(pred_filename+"gt", pred_filename, "data/cosql/database/", etype, self.kmaps, False, False, False)

            results = evaluate(gold_filename, pred_filename, self.db_dir, etype, self.kmaps, False, False, False)
            sys.stdout = old_print
            of.close()
        return results
