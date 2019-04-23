def get_base_env(env):
    """ removes all wrappers   """
    while hasattr(env, 'env'):
        env = env.env
    return env


def copy_atari_state(env):
    env = get_base_env(env)
    return env.clone_full_state()


def restore_atari_state(env, snapshot):
    env = get_base_env(env)
    env.restore_full_state(snapshot)


def is_atari_game(env):
    """ Verify whether game uses the Arcade Learning Environment   """
    env = get_base_env(env)
    return hasattr(env, 'ale')