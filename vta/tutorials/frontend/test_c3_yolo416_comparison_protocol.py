from copy import deepcopy
from collections import Counter
import unittest
from c3_yolo416_comparison_protocol import manifest,validate,analyse,SEEDS


def pool(n=11):
    return [{'object_id':'node%03d'%i,'kind':'operator','programs':{
        'TH':{'execution_identity':{'code':'th'+str(i)}},
        'P':{'execution_identity':{'code':'p'+str(i)}}}} for i in range(n)]+[
        {'object_id':'whole_graph','kind':'graph','programs':{
            'TH':{'execution_identity':{'code':'th_graph'}},
            'P':{'execution_identity':{'code':'p_graph'}}}}]


class LedgerTests(unittest.TestCase):
    def test_full_pool_budget_and_seed_rounds(self):
        m=manifest('attempt',pool());self.assertEqual(len(m['calls']),1440)
        self.assertEqual(len(m['sessions']),240);self.assertEqual(m['formal_samples'],432)
        self.assertEqual(len(m['calls']),m['total_target_calls'])
        self.assertTrue(all(s['target_calls']==6 for s in m['sessions']))
        counts=Counter((r['program_key'],r['round'],r['seed']) for r in m['calls'] if r['timed'])
        self.assertEqual(set(counts.values()),{1})
        self.assertEqual(len(counts),24*6*3)
        for s in m['sessions']:
            rows=[r for r in m['calls'] if r['session_id']==s['session_id']]
            self.assertEqual([r['position'] for r in rows],list(range(1,7)))

    def test_round_order_and_warmup_prefix(self):
        m=manifest('a',pool(3))
        for r in range(1,7):
            sessions=[s for s in m['sessions'] if s['round']==r]
            grouped=[s['object_id'] for s in sessions[::2]]
            ids=['node000','node001','node002']
            off=(r-1)%3
            self.assertEqual(grouped,(ids[off:]+ids[:off] if r<6 else ids[::-1])+['whole_graph'])
            self.assertEqual([s['variant'] for s in sessions[:2]],['TH','P'] if r%2 else ['P','TH'])
            for s in sessions:
                rows=[x for x in m['calls'] if x['session_id']==s['session_id']]
                self.assertEqual([x['seed'] for x in rows[:3]],[x['seed'] for x in rows[3:]])
                self.assertEqual([x['phase'] for x in rows],['warmup']*3+['formal']*3)

    def test_structural_alias_deduplicates_without_fake_gain(self):
        p=pool(0);p[0]['programs']['P']=deepcopy(p[0]['programs']['TH'])
        p[0]['representatives']={'TH':'TH','P':'TH'}
        m=manifest('a',p);self.assertEqual(m['total_target_calls'],60)
        results=[dict(x,status='PASS',target_calls=1,elapsed_ms=1.) for x in m['calls']]
        a=analyse(m,results)['objects'][0]
        self.assertNotIn('Gain_P_over_TH_percent',a)
        p[0]['programs']['P']['execution_identity']['code']='different'
        with self.assertRaises(ValueError):manifest('a',p)

    def test_reject_missing_duplicate_changed_identity_and_bad_time(self):
        m=manifest('a',pool(0));good=[dict(x,status='PASS',target_calls=1,elapsed_ms=2.) for x in m['calls']]
        self.assertTrue(validate(m,good))
        cases=[good[:-1],good+[good[-1]]]
        for k,v in [('elapsed_ms',float('nan')),('elapsed_ms',True),('target_calls',2),('variant','X')]:
            rows=deepcopy(good);rows[0][k]=v;cases.append(rows)
        for rows in cases:
            with self.assertRaises(ValueError):validate(m,rows)

    def test_gain_is_median_of_paired_ratios(self):
        m=manifest('a',pool(0));rows=[]
        th=[1,2,3,4,5,100];p=[.9,1,4,3,6,80]
        for x in m['calls']:
            elapsed=(th if x['variant']=='TH' else p)[(x['round'] or 1)-1]
            rows.append(dict(x,status='PASS',target_calls=1,elapsed_ms=elapsed))
        result=analyse(m,rows)['objects'][0]
        from statistics import median
        self.assertEqual(result['Gain_P_over_TH_percent'],median(100*(1-b/a) for a,b in zip(th,p)))
        self.assertNotEqual(result['Gain_P_over_TH_percent'],100*(1-median(p)/median(th)))


if __name__=='__main__':unittest.main()
