class DB:
    """
    An abstract base class for database connections.
    """

    def __init__(self, database_url: str):
        self.database_url = database_url

