# import torch
# from typing import List

# class MultiSchedulerWrapper:
#     def __init__(self, optimizer., schedulers, switch_step):

#         self.optimizer = optimizer
#         self.schedulers = schedulers
#         self.switch_step = switch_step
#         self.current_scheduler_index = 0

#     def step(self, global_step):
#         if global_step >= self.switch_step and self.current_scheduler_index == 0:
#             self.current_scheduler_index = 1
#             print(f"Switched to scheduler {self.current_scheduler_index} at step {global_step}")

#         self.schedulers[self.current_scheduler_index].step()

#     def get_last_lr(self):
#         return self.schedulers[self.current_scheduler_index].get_last_lr()