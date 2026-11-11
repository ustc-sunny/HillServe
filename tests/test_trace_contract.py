import unittest
from hybrid_eval.trace_contract import validate
class TraceTests(unittest.TestCase):
 def base(self):return {'method_is_confirmed':False,'future_query_visibility':'dependency_gated','payload_bytes':4096}
 def events(self):return [{'id':'append','request':1,'layer':0,'original_position':0,'deps':[],'append':True,'data_refs':[{'file':'owned.bin','offset':0,'length':2048}]},{'id':'q','request':1,'layer':0,'original_position':0,'deps':['append'],'query':True}]
 def test_diagnostic(self):self.assertFalse(validate(self.base(),self.events(),'fixed_request_ids')['method_execution_validated'])
 def test_unbound(self):
  with self.assertRaises(ValueError):validate(self.base(),self.events(),'recompute_scores')
 def test_oracle_leak(self):
  m=self.base();m.update(method_is_confirmed=True,policy_hash='x',oracle_ids_available_to_policy=True)
  with self.assertRaises(ValueError):validate(m,self.events(),'recompute_scores')
 def test_future(self):
  with self.assertRaises(ValueError):validate(self.base(),self.events()[::-1],'fixed_request_ids')
 def test_duplicate_append(self):
  e=self.events();x=dict(e[0]);x['id']='append2';e.append(x)
  with self.assertRaises(ValueError):validate(self.base(),e,'fixed_request_ids')
 def test_slice(self):
  e=self.events();e[0]['data_refs'][0]['length']=8192
  with self.assertRaises(ValueError):validate(self.base(),e,'fixed_request_ids')
 def test_path(self):
  e=self.events();e[0]['data_refs'][0]['file']='../outside'
  with self.assertRaises(ValueError):validate(self.base(),e,'fixed_request_ids')
