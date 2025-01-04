# allocation successful, but the pointer cannot be used by pytorch because they are in different cuda context

from cuda.bindings import driver

# allocate ptr
result, ptr = driver.cuMemAddressReserve(size, 0, 0, 0)
assert result.value == 0

# allocate the handle
prop = driver.CUmemAllocationProp()
prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
prop.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
prop.location.id = torch.cuda.current_device()
prop.win32HandleMetaData = 0 # this is very critical, cannot use ctypes.c_void_p(0)
result, handle = driver.cuMemCreate(size, prop, 0)
assert result.value == 0

# map the memory
result, = driver.cuMemMap(ptr, size, 0, handle, 0)
assert result.value == 0

# set access
access = driver.CUmemAccessDesc()
access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
access.location.id = torch.cuda.current_device()
access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
result, = driver.cuMemSetAccess(ptr, size, [access], 1)
assert result.value == 0

print(f"Python side: Malloc called with size={size}, ptr=0x{ptr:x}")
return ptr.value
