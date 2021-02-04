import multiprocessing
from src import retro_util as retro_util
import gc

MAKE_RETRIES = 5


def set_retro_make(new_retro_make_func):
    RetroWrapper.retro_make_func = new_retro_make_func


def _retrocom(rx, tx, game, kwargs):
    """
    This function is the target for RetroWrapper's internal
    process and does all the work of communicating with the
    environment.
    """
    env = RetroWrapper.retro_make_func(game, **kwargs)

    # Sit around on the queue, waiting for calls from RetroWrapper
    while True:
        attr, args, kwargs = rx.get()

        # First, handle special case where the wrapper is asking if attr is callable.
        # In this case, we actually have RetroWrapper.symbol, attr, and {}.
        if attr == RetroWrapper.symbol:
            result = env.__getattribute__(args)
            tx.put(callable(result))
        elif attr == "close":
            env.close()
            break
        else:
            # Otherwise, handle the request
            result = getattr(env, attr)
            if callable(result):
                result = result(*args, **kwargs)
            tx.put(result)


# 使用这个可以创建多个retro游戏
class RetroWrapper():
    """
    This class is a thin wrapper around a retro environment.
    The purpose of this class is to protect us from the fact
    that each Python process can only have a single retro
    environment at a time, and we would like potentially
    several.
    This class gets around this limitation by spawning a process
    internally that sits around waiting for retro environment
    API calls, asking its own local copy of the environment, and
    then returning the answer.
    Call functions on this object exactly as if it were a retro env.
    """
    symbol = "THIS IS A SPECIAL MESSAGE FOR YOU"
    retro_make_func = retro_util.retro_make_func

    def __init__(self, game, **kwargs):
        tempenv = None
        retry_counter = MAKE_RETRIES
        while True:
            try:
                tempenv = RetroWrapper.retro_make_func(game, **kwargs)
            except RuntimeError:  # Sometimes we need to gc.collect because previous tempenvs haven't been cleaned up.
                gc.collect()
                retry_counter -= 1
                if retry_counter > 0:
                    continue
            break

        if tempenv == None:
            raise RuntimeError('Unable to create tempenv')

        tempenv.reset()

        if hasattr(tempenv, 'unwrapped'):  # Wrappers don't have gamename or initial_state
            tempenv_unwrapped = tempenv.unwrapped
            self.gamename = tempenv_unwrapped.gamename
            self.initial_state = tempenv_unwrapped.initial_state

        self.action_space = tempenv.action_space
        self.metadata = tempenv.metadata
        self.observation_space = tempenv.observation_space
        self.reward_range = tempenv.reward_range
        tempenv.close()

        self._rx = multiprocessing.Queue()
        self._tx = multiprocessing.Queue()
        self._proc = multiprocessing.Process(target=_retrocom, args=(self._tx, self._rx, game, kwargs), daemon=True)
        self._proc.start()

    def __del__(self):
        """
        Make sure to clean up.
        """
        self.close()

    def __getattr__(self, attr):
        """
        Any time a client calls anything on our object, we want to check to
        see if we can answer without having to ask the retro process. Usually,
        we will have to ask it. If we do, we put a request into the queue for the
        result of whatever the client requested and block until it comes back.
        Otherwise we simply give the client whatever we have that they want.
        BTW: This doesn't work for magic methods. To get those working is a little more involved. TODO
        """
        # E.g.: Client calls env.step(action)
        ignore_list = ['class', 'mro', 'new', 'init', 'setattr', 'getattr', 'getattribute']
        if attr in self.__dict__ and attr not in ignore_list:
            # 1. Check if we have a step function. If so, return it.
            return attr
        else:
            # 2. If we don't, return a function that calls step with whatever args are passed in to it.
            is_callable = self._ask_if_attr_is_callable(attr)

            if is_callable:
                # The result of getattr(attr) is a callable, so return a wrapper
                # that pretends to be the function the user was trying to call
                def wrapper(*args, **kwargs):
                    self._tx.put((attr, args, kwargs))
                    return self._rx.get()

                return wrapper
            else:
                # The result of getattr(attr) is not a callable, so we should just
                # execute the request for the user and return the result
                self._tx.put((attr, [], {}))
                return self._tx.get()

    def _ask_if_attr_is_callable(self, attr):
        """
        Returns whether or not the attribute is a callable.
        """
        self._tx.put((RetroWrapper.symbol, attr, {}))
        return self._rx.get()

    def close(self):
        """
        Shutdown the environment.
        """
        if "_tx" in self.__dict__ and "_proc" in self.__dict__:
            self._tx.put(("close", (), {}))
            self._proc.join()
