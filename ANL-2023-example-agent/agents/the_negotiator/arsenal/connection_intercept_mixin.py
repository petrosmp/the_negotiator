from geniusweb.party.DefaultParty import DefaultParty

class CustomConnection:
    """
    A mock connection that implements all the methods of the real deal (see
    geniusweb.connection.ConnectionEnd.ConnectionEnd) but really just sends
    things to a proxy.    
    """

    def __init__(self, proxy) -> None:
        self._proxy: DefaultParty = proxy
    
    def send(self, data):
        '''
		Send data out (and flush the output so that there are no excessive delays
		in sending the data). This call is assumed to return immediately (never
		block, eg on synchronized, Thread.sleep, IO, etc). When this is called
		multiple times in sequence, the data should arrive at the receiver end in
		the same order.
		
		@param data the data to be sent.
		@throws ConnectionError if the data failed to be sent.
		'''

        self._proxy.set_connection_data(data)


    def getReference(self):
        pass

    def getRemoteURI(self): 
        pass

    def close(self):
        pass

    def getError(self):
        pass





class ConnectionInterceptMixin:
    """
    A Mixin to override the getConnection() method of any agent that uses it
    and redirect it to a "proxy" class.
    """

    def set_proxy(self, proxy: DefaultParty):
        """Set the given party as a proxy"""

        self._proxy = proxy


    def getConnection(self):
        """Return an object that will redirect anything being sent to the connection to the proxy agent"""

        try:
            return self._proxy_connection
        except AttributeError:
            self._proxy_connection = CustomConnection(self._proxy)
            return self._proxy_connection
 
