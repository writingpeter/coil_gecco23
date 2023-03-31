# scheduler used by optimise and comparison GA
# takes in a list of requests of the format [{'id': i, 'duration': duration, 'scheduled': False, 'robot_id': None}, ...]
# takes in a list of robots of the format [{'id': robot_id, 'original_duration': duration, 'remaining_duration': duration}, ...]
# outputs the number of requests met by robots and updated requests and robots lists
def scheduler(requests, robots, verbose=False):
    DELTA = 10 # set a small amount of wiggle room to allow for a close enough approximate match
    total_requests_met = 0

    for i, request in enumerate(requests):
        hypothetical_remaining_duration = []

        # find out the remaining durations for each robot, if they were to meet the request
        for j, robot in enumerate(robots):
            remaining_duration = robot['remaining_duration'] - request['duration']
            hypothetical_remaining_duration.append({'robot_id': j, 'remaining_duration': remaining_duration})
        sorted_hypothetical_remaining_duration = sorted(hypothetical_remaining_duration, key=lambda d: d['remaining_duration']) 

        # if the lowest remaining duration is positive and less than 10, use it
        lowest = sorted_hypothetical_remaining_duration[0]
        if lowest['remaining_duration'] > 0 and lowest['remaining_duration'] < DELTA:
            robots[lowest['robot_id']]['remaining_duration'] = lowest['remaining_duration']
            request['scheduled'] = True
            request['robot_id'] = lowest['robot_id']
            total_requests_met += 1
        else: # otherwise use the robot with the highest remaining duration after meeting the request
            highest = sorted_hypothetical_remaining_duration[-1]
            if highest['remaining_duration'] > 0:
                robots[highest['robot_id']]['remaining_duration'] = highest['remaining_duration']
                request['scheduled'] = True
                request['robot_id'] = highest['robot_id']
                total_requests_met += 1

    result = {'total_requests_met': total_requests_met}
    result['robots'] = robots

    if verbose:
        result['requests'] = requests
        # result['robots'] = robots

    return result