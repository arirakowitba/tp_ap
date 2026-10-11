"""Small deterministic workflow checks; never trains a model or accesses a network."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
import finalize_round4 as f


class FrozenWorkflow(unittest.TestCase):
    def test_selection_confirmation_and_refit_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache, out = root/'selection', root/'out'
            cache.mkdir(); out.mkdir()
            rows = np.arange(21000)
            y = rows % 2
            good = np.where(y, .99, .01)
            prior = good.copy(); prior[0] = .99
            fp = dict(train_sha256='train-fixture', seed=42, holdout_folds=3)
            pd.DataFrame(dict(row_index=rows, y_used=y)).to_csv(cache/'evaluation_rows.csv', index=False)
            (cache/'protocol.json').write_text(json.dumps(dict(fingerprint=fp)))
            prior_path = root/'prior.csv'
            pd.DataFrame(dict(row_index=rows, y_used=y, probability_used=prior)).to_csv(prior_path,index=False)
            ablation = out/'fixed_ablation'; ablation.mkdir()
            cfg = dict(name='r4_lgb_mock', family='lgb', params=dict(n_estimators=1), use_media=True)
            (ablation/'r4_lgb_mock.json').write_text(json.dumps(cfg))
            pred = ablation/'r4_lgb_mock_predictions.csv'
            pd.DataFrame(dict(row_index=rows[::-1], y_used=y[::-1], probability_used=good[::-1])).to_csv(pred,index=False)
            args = SimpleNamespace(output=str(out),cache=str(cache),prior_selection=str(prior_path),candidates=str(out))
            frozen = f.select(args)
            self.assertEqual(frozen['selection_metrics']['accuracy'],1)
            self.assertEqual(f.load_frozen(out/'round4_selected.json')['selection_digest'],frozen['selection_digest'])
            with self.assertRaises(FileExistsError): f.select(args)
            bad = pd.read_csv(pred); bad.loc[0,'y_used'] = 1-bad.loc[0,'y_used']; bad.to_csv(pred,index=False)
            with self.assertRaises(ValueError): f.aligned_predictions(pred,rows,y)
            confirm = root/'confirm'; confirm.mkdir()
            cr = rows[:14000][::-1]; cy = y[:14000][::-1]
            pd.DataFrame(dict(row_index=cr,y_used=cy)).to_csv(confirm/'evaluation_rows.csv',index=False)
            (confirm/'protocol.json').write_text(json.dumps(dict(fingerprint=dict(fp,seed=731,holdout_folds=2))))
            cp = root/'prior-confirm.csv'
            pd.DataFrame(dict(row_index=rows[:14000],y_used=y[:14000],probability_used=prior[:14000])).to_csv(cp,index=False)
            args.selected=str(out/'round4_selected.json');args.cache=str(confirm);args.prior_confirmation=str(cp)
            with patch.object(f,'fit_component',return_value=good[:14000][::-1]):
                result=f.confirm(args)
            self.assertTrue(result['promote'])
            self.assertEqual(result['paired']['net_correct'],1)
            self.assertEqual(result['selection_confirmation_overlap'],14000)
            # Promotion is not a claim that a new public score has been measured.
            self.assertIsNone(result['public_accuracy'])
            args.confirmation=str(out/'round4_confirmation.json')
            full=root/'full'; full.mkdir(); args.cache=str(full)
            full_fp=dict(train_sha256='train-fixture',test_sha256='test-fixture',full=True)
            (full/'protocol.json').write_text(json.dumps(dict(fingerprint=full_fp)))
            refmeta=root/'meta.json'; refmeta.write_text(json.dumps(dict(train_sha256='train-fixture',test_sha256='test-fixture')))
            args.prior_metadata=str(refmeta)
            ids=np.arange(1,30001)
            pd.to_pickle(dict(tr=pd.DataFrame(index=range(70000)),va=pd.DataFrame(index=range(30000)),t=np.arange(70000),v=np.arange(30000),y_eval=None,test_ids=ids,fingerprint=full_fp),full/'native.pkl')
            ref=root/'prior-full.csv'
            pd.DataFrame(dict(ID=ids,probability_used=np.full(30000,.1))).to_csv(ref,index=False)
            args.prior_full=str(ref);args.submission=str(root/'submission.csv')
            with patch.object(f,'fit_component',return_value=np.full(30000,.9)):
                fitted=f.refit(args)
            self.assertEqual(fitted['rows'],30000)
            self.assertEqual(list(pd.read_csv(args.submission)),['ID','condition'])
            badfp=dict(full_fp,test_sha256='wrong')
            (full/'protocol.json').write_text(json.dumps(dict(fingerprint=badfp)))
            with self.assertRaises(ValueError):f.refit(args)
            corrupted=dict(frozen,threshold=.3)
            (out/'round4_selected.json').write_text(json.dumps(corrupted))
            with self.assertRaises(ValueError):f.load_frozen(out/'round4_selected.json')


if __name__=='__main__':
    unittest.main()
