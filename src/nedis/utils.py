import re
import unicodedata


def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    Source: https://stackoverflow.com/a/295466/991496
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def select(name, include=None, exclude=None):
    """
    Checks whether name is selected based on an include and an exclude filter.
    If include is defined it is applied first. 
    Exclude is applied only if the name is included.
    """

    # filter pipelines
    if include is not None:
        if isinstance(include, str):
            if re.match(include, name) is None:
                return False
        elif name not in include:
            return False

    if exclude is not None:
        if isinstance(exclude, str):
            if re.match(exclude, name) is not None:
                return False
        elif name in exclude:
            return False

    return True