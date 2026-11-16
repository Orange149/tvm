"""Frozen three-way counts, balance, path identity and statistics gates."""
from collections import Counter
import copy
import unittest
from c3_fallback_identity_protocol import manifest, validate, analyse, transport_contract, ROUND_ORDERS, SEEDS


def pool():
    return [{'object_id':'yolo224_node%d'%n,'kind':'operator',
             'programs':{v:{'module':v+str(n)} for v in ('F','T','U')}} for n in (34,46)]


def results(ledger):
    return [{**r,'status':'PASS','target_calls':1,
             'elapsed_ms':None if r['phase'].startswith('diagnostic_') else {'F':100.,'T':50.,'U':45.}[r['variant']]}
            for r in ledger['calls']]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.ledger=manifest('zero-target-tool-test',pool())

    def test_exact_budget_and_once_per_seed(self):
        m=self.ledger
        self.assertEqual((len(m['calls']),len(m['sessions']),m['formal_samples']),(360,60,108))
        self.assertEqual(set(Counter((r['object_id'],r['variant']) for r in m['calls']).values()),{60})
        self.assertEqual(set(Counter((r['object_id'],r['variant'],r['round'],r['seed'])
                                   for r in m['calls'] if r['timed']).values()),{1})
        self.assertTrue(all(s['target_calls']==6 for s in m['sessions']))

    def test_balanced_three_way_order_and_seed_rotations(self):
        for node in (34,46):
            oid='yolo224_node%d'%node
            for r in range(1,7):
                sessions=[s for s in self.ledger['sessions'] if s['round']==r and s['object_id']==oid]
                self.assertEqual(tuple(s['variant'] for s in sessions),ROUND_ORDERS[r-1])
                for session in sessions:
                    seeds=[row['seed'] for row in self.ledger['calls'] if row['session_id']==session['session_id']]
                    rotation=(r-1)%3
                    self.assertEqual(seeds,list(SEEDS[rotation:]+SEEDS[:rotation])*2)
        self.assertEqual(Counter((p,v) for order in ROUND_ORDERS for p,v in enumerate(order)),
                         Counter({(p,v):2 for p in range(3) for v in ('F','T','U')}))

    def test_complete_three_pair_statistics(self):
        report=analyse(self.ledger,results(self.ledger))
        for item in report['objects']:
            self.assertAlmostEqual(item['pairwise']['T/F']['Gain_percent'],50.)
            self.assertAlmostEqual(item['pairwise']['U/T']['Gain_percent'],10.)
            self.assertAlmostEqual(item['pairwise']['U/F']['Gain_percent'],55.)
            self.assertEqual(item['pairwise']['U/T']['strict_wins'],6)

    def test_missing_duplicate_or_changed_identity_rejected(self):
        rows=results(self.ledger)
        for changed in (rows[:-1], rows+[rows[-1]], [dict(rows[0],variant='T')]+rows[1:]):
            with self.assertRaises(ValueError):validate(self.ledger,changed)
        changed=copy.deepcopy(self.ledger);changed['calls'][0]['program_identity']['module']='changed'
        with self.assertRaises(ValueError):validate(changed,rows)

    def test_nonfinite_and_hidden_calls_rejected(self):
        for elapsed in (float('nan'),float('inf'),0.,-1.,True):
            rows=results(self.ledger);rows[-1]['elapsed_ms']=elapsed
            with self.assertRaises(ValueError):validate(self.ledger,rows)
        rows=results(self.ledger);rows[0]['target_calls']=2
        with self.assertRaises(ValueError):validate(self.ledger,rows)

    def test_explicit_labels_not_tophub_and_fixed_transport(self):
        wrong=pool();wrong[0]['programs']['TH']=wrong[0]['programs'].pop('F')
        with self.assertRaises(ValueError):manifest('x',wrong)
        t=transport_contract(self.ledger)
        self.assertEqual(len({x['port'] for x in t['sessions']}),60)
        self.assertEqual((t['target_retries'],t['rpc_connect_attempts_per_session']),(0,1))
        with self.assertRaises(ValueError):transport_contract(self.ledger,base_port=0)


if __name__=='__main__':unittest.main()
