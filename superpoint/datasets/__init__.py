def get_dataset(name):
    mod = __import__('superpoint.datasets.{}'.format(name), fromlist=[''])
    return getattr(mod, _module_to_class(name))


def _module_to_class(name):
    # n.capitalize() will only capital the first character of a word
    return ''.join(n.capitalize() for n in name.split('_'))
