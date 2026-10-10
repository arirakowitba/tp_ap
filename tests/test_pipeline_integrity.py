import json,sys,tempfile,unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from competition_pipeline_v2 import audit,load_data,groups_for
from optimize_leaderboard import split
from ap2026q2_advanced_optimized import _attributes
class IntegrityAndLeakageTests(unittest.TestCase):
    def test_bad_json_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.jsonlines';path.write_text('{"condition":"used"}\n{"condition"')
            result=audit(path,2);self.assertFalse(result['complete']);self.assertEqual(result['valid_rows'],1)
            self.assertEqual(result['invalid_lines'][0]['line'],2)
    def test_family_split_keeps_siblings_together(self):
        raw=pd.DataFrame({'id':[str(i) for i in range(120)],'parent_item_id':[f'f{i//3}' for i in range(120)],'condition':['new' if i//3%2 else 'used' for i in range(120)]})
        y=(raw.condition=='used').to_numpy();g=groups_for(raw);t,v=split(y,g,42,2)
        self.assertFalse(set(g[t])&set(g[v]));self.assertEqual(len(t)+len(v),len(raw))
    def test_explicit_target_attribute_is_excluded(self):
        raw=pd.DataFrame({'attributes':[[{'id':'ITEM_CONDITION','name':'Condición','value_name':'Nuevo'},{'id':'BRAND','name':'Marca','value_name':'Acme'}]]})
        features,ids=_attributes(raw,top_k=250);self.assertEqual(ids,['BRAND']);self.assertEqual(features.columns.tolist(),['attr_BRAND'])
    def test_missing_parent_uses_individual_key(self):
        raw=pd.DataFrame({'id':['a','b'],'parent_item_id':[None,None]});self.assertEqual(len(set(groups_for(raw))),2)
if __name__=='__main__':unittest.main()
