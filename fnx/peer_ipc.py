"""Bounded device IPC buffers; publishers synchronize before sending metadata.

Only the hidden tensor payload uses device-to-device copies. Commands/acks stay
on the existing process pipe. The owner must outlive all borrowers.
"""
import acl
import torch


def checked(ret, operation):
    if ret != 0:
        raise RuntimeError(f'{operation}: ACL error {ret}')


def process_id():
    pid, ret = acl.rt.device_get_bare_tgid()
    checked(ret, 'device_get_bare_tgid')
    return pid


def copy_device(dst, src, size):
    stream = torch.npu.current_stream().npu_stream
    checked(acl.rt.memcpy_async(int(dst), size, int(src), size, 3, stream),
            'memcpy_async DEVICE_TO_DEVICE')


class Owner:
    def __init__(self, slot_bytes, slots=2):
        self.slot_bytes, self.slots = int(slot_bytes), int(slots)
        self.ptr = None
        self.key = None
        ptr, ret = acl.rt.malloc(self.slot_bytes * self.slots, 3)
        checked(ret, 'malloc P2P')
        self.ptr = ptr
        try:
            key, ret = acl.rt.ipc_mem_get_export_key(ptr, self.slot_bytes*self.slots, 65, 0)
            checked(ret, 'ipc_mem_get_export_key')
            self.key = key
        except BaseException:
            acl.rt.free(ptr)
            self.ptr = None
            raise

    def allow(self, pid):
        checked(acl.rt.ipc_mem_set_import_pid(self.key, [int(pid)]), 'ipc_mem_set_import_pid')

    def descriptor(self):
        return {'key': self.key, 'slot_bytes': self.slot_bytes, 'slots': self.slots}

    def publish(self, tensor, slot):
        nbytes = tensor.numel()*tensor.element_size()
        if not tensor.is_contiguous() or not 0 <= slot < self.slots or nbytes > self.slot_bytes:
            raise ValueError('Invalid peer payload or slot')
        # Captured producer graphs may use another stream. Publish only after
        # all producer writes complete; the process pipe is the cross-device fence.
        torch.npu.synchronize()
        copy_device(self.ptr+slot*self.slot_bytes, tensor.data_ptr(), nbytes)
        torch.npu.current_stream().synchronize()

    def close(self):
        if self.key is not None:
            checked(acl.rt.ipc_mem_close(self.key), 'owner ipc_mem_close')
            self.key = None
        if self.ptr is not None:
            checked(acl.rt.free(self.ptr), 'owner free')
            self.ptr = None


class Borrower:
    def __init__(self, descriptor):
        self.key = descriptor['key']
        self.slot_bytes = int(descriptor['slot_bytes'])
        self.slots = int(descriptor['slots'])
        ptr, ret = acl.rt.ipc_mem_import_by_key(self.key, 1)
        checked(ret, 'ipc_mem_import_by_key ENABLE_PEER_ACCESS')
        self.ptr = ptr

    def receive(self, tensor, slot):
        nbytes = tensor.numel()*tensor.element_size()
        if not tensor.is_contiguous() or not 0 <= slot < self.slots or nbytes > self.slot_bytes:
            raise ValueError('Invalid peer destination or slot')
        copy_device(tensor.data_ptr(), self.ptr+slot*self.slot_bytes, nbytes)
        torch.npu.current_stream().synchronize()

    def close(self):
        if self.ptr is not None:
            checked(acl.rt.ipc_mem_close(self.key), 'borrower ipc_mem_close')
            self.ptr = None
