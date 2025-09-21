You should implement logging in this project by utilizing the DynamicLogger package, which is useful to reduce changing code files in order to turn on or off some log calls or adjust their other logging options. and we mostly just need to change highOrderOptions at `utils/dyLog_highOrderOptions.py`. so I provide minimal how to use it.
one of the other motives that we use loggings described as below is not to have crowded log files making it easier for to shape understanding while debugging, with only enabling the log calls from funcs and methods which have importance, in doing that particular task.

- I will place the main dynamicLogger project at `utils\dynamicLoggerModule` and u should never touch it.
- I will place `/utils/loggerInstance.py`, which has instances of the uniLogger, uniDebugLogger. and highOrderOptions is at `utils/dyLog_highOrderOptions.py`.
- u don't need to touch `/utils/loggerInstance.py` at all and we are going only to modify highOrderOptions at `utils/dyLog_highOrderOptions.py`

- uniLogger, uniDebugLogger for logging and prints, so if u see prints or logging with other types replace them uniLogger, uniDebugLogger
- almost to all files which are going to use logging add this import `from utils.loggerInstance import uniLogger, uniDebugLogger`
- `info`,`warning`,`error` methods, which can be called on uniLogger, like `uniLogger.info(msg)`, `uniLogger.warning(msg)`, `uniLogger.error(msg)`
- `debug` method can be called on uniDebugLogger, like `uniDebugLogger.debug(msg)`
- logs of uniDebugLogger are disabled by default, and in order to enable them we should add them by changing highOrderOptions(by adding `funcName` or `class.method` dicts with `{"exclude":False}`). this way we selectively activate debug logs, facing less crowded logs.
- consider UX and debugging while u think about whether to choose uniLogger or uniDebugLogger. if log line requires to convey infos, warnings and errors to final user, choose uniLogger, and if it provides useful info about debugging or developing the project and final user doesn't require to see that choose `uniDebugLogger.debug(msg)`
- in writing/rewriting methods and funcs, try not to be too pedantic but consider which infos are useful for final user and which debug infos are going to be useful. all in all, try to provide adequate info whether with uniLogger or uniDebugLogger
- this is very very important that now on with any request, or especially in cases we have bug, u should give the code of `utils/dyLog_highOrderOptions.py` modifying highOrderOptions, in order to include debug logs from funcs and methods. note if u enable too many funcs/methods u make logs too crowded, making debugging harder and if u don't enable enough we don't get necessary debug info, so consider related funcs and methods to the procedure we are working on and enable most important funcs and methods. ofc don't worry we can still modify highOrderOptions later.
- this is very very important that especially in cases we have bug, consider adding more debug lines to that func/method with `uniDebugLogger.debug(msg)`, also don't forget to enable it in highOrderOptions.
- below is an example of how u should do these, and if u have any questions, ask me.

```utils/dyLog_highOrderOptions.py
globalDebugPrint = True
highOrderOptions = {
    # i.e. to enable debug logs in a method (for i.e. setAttrsMethod from MainComponentClass) 
    # "MainComponentClass.setAttrsMethod": {"exclude":False},
    # i.e. to enable debug logs in some function
    # "funcA": {"exclude":False},
    "BankAccount.withdraw": {"exclude": False},
    "calculateDiscount": {"exclude": False},
}
```
```
from utils.loggerInstance import uniLogger, uniDebugLogger
class BankAccount:
    def __init__(self, balance):
        self.balance = balance
    def withdraw(self, amount):
        if amount > self.balance:
            uniLogger.error('Insufficient funds')  # User important info
            uniDebugLogger.debug(f"Tried to withdraw {amount}, balance: {self.balance}")  # Internal debug
            return False
        self.balance -= amount
        return True
def calculateDiscount(price, discountRate):
    finalPrice = price * (1 - discountRate)
    if discountRate > 0.5:
        uniLogger.warning("Large discount applied")  # User important info
    debugInfo = f"Input - price: {price}, discountRate: {discountRate}"  # Internal
    uniDebugLogger.debug(f"{debugInfo}, finalPrice: {finalPrice}")  # Internal debug
    return finalPrice
```
also to test funcs add
```
    def testMdToHtmlBasicFormatting(self):
        uniDebugLogger.debug("Running")
        ...
        uniDebugLogger.debug("succeed")
```
do u know the importance of highOrderOptions? putting too many funcs/methods in it or putting funcs which are called so many times or log long text, would make debugging from log files harder and not putting enough methods would deprive us from having those logs so update highOrderOptions and even add more debugging lines to the code to related funcs/methods