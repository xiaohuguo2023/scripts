def get_new_pid(current_pid, num_sms):
    # Number of XCDs
    num_xcds = 8
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = num_sms // num_xcds
    # Compute current XCD and local pid within the XCD
    xcd = current_pid % num_xcds
    local_pid = current_pid // num_xcds

    # Calculate new pid based on the new grouping
    new_pid = xcd * pids_per_xcd + local_pid
    return new_pid

# Testing the function and printing new pids for each XCD
num_sms = 608
num_xcds = 8
pids_per_xcd = num_sms // num_xcds
for xcd in range(8):
    print(f"\nXCD {xcd}:")
    for local_pid in range(pids_per_xcd):
        current_pid = xcd + local_pid * num_xcds
        if current_pid < num_sms:  # To ensure we don't exceed the range
            print(f"Current pid: {current_pid} -> New pid: {get_new_pid(current_pid, num_sms)}")
