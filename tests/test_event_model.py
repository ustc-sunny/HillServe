import unittest
from hybrid_eval.event_model import Task,schedule,from_profile,require_publication_validation
class ModelTests(unittest.TestCase):
 def test_serial(self):self.assertEqual(schedule([Task('a',2,{'cpu':1}),Task('b',3,{'cpu':1},('a',))],{'cpu':1})['target_time_ms'],5)
 def test_parallel(self):self.assertEqual(schedule([Task('a',2,{'cpu':1}),Task('b',3,{'gpu':1})],{'cpu':1,'gpu':1})['target_time_ms'],3)
 def test_shared_pcie(self):self.assertEqual(schedule([Task('a',2,{'pcie':1}),Task('b',3,{'pcie':1})],{'pcie':1})['target_time_ms'],5)
 def test_worker_capacity(self):self.assertEqual(schedule([Task(str(i),2,{'cpu':1}) for i in range(3)],{'cpu':2})['target_time_ms'],4)
 def test_css_barrier(self):
  r=schedule([Task('host',2),Task('cold',4),Task('select',1,deps=('host','cold'))],{})
  self.assertEqual(r['target_time_ms'],5)
 def test_future_query(self):
  r=schedule([Task('attention',3),Task('next_q',2,deps=('attention',)),Task('score',1,deps=('next_q',))],{})
  self.assertEqual(r['timeline'][-1]['start_ms'],5)
 def test_buffer_backpressure(self):
  r=schedule([Task('p1',1,acquire={'b':1}),Task('p2',1,acquire={'b':1}),Task('c1',2,deps=('p1',),release={'b':1}),Task('c2',2,deps=('p2',),release={'b':1})],{}, {'b':1})
  self.assertEqual(r['target_time_ms'],6)
 def test_deadlock(self):
  with self.assertRaises(ValueError):schedule([Task('a',1,deps=('b',)),Task('b',1,deps=('a',))],{})
 def test_release(self):
  with self.assertRaises(ValueError):schedule([Task('a',1,release={'b':1})],{}, {'b':1})
 def test_missing_query(self):
  with self.assertRaises(ValueError):schedule([Task('a',1,deps=('future',))],{})
 def test_profile_no_fabrication(self):
  with self.assertRaises(ValueError):from_profile('a',{'status':'passed','evidence_mode':'M_COMPONENT','service_ms':None,'includes_queue_wait':False})
 def test_profile_wait(self):
  with self.assertRaises(ValueError):from_profile('a',{'status':'passed','evidence_mode':'M_COMPONENT','service_ms':2,'includes_queue_wait':True})
 def test_profile(self):
  t=from_profile('a',{'profile_id':'measured-1','status':'passed','evidence_mode':'M_COMPONENT','service_ms':2,'includes_queue_wait':False})
  self.assertEqual(schedule([t],{})['target_time_ms'],2)
 def test_no_claim(self):
  with self.assertRaises(ValueError):require_publication_validation({})
 def test_holdout_leak(self):
  with self.assertRaises(ValueError):require_publication_validation({'independent_holdout_passed':True,'measurement_ids':['a'],'holdout_ids':['a']})
 def test_invalid_time(self):
  with self.assertRaises(ValueError):schedule([Task('a',float('nan'))],{})
